# coding=utf-8
import mxnet as mx
from mxnet import gluon
import cv2 
import sys
from postprocess import pse_poster
from model.net import PSENet
import glob
import os
from mxnet.gluon.data.vision import transforms
import numpy as np
import time
import logging
logging.basicConfig(level=logging.INFO)

def resize_image(im, max_side_len=1200):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.

    #ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w


    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 + 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 + 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


"""
References 
1. https://github.com/liuheng92/tensorflow_PSENet
2. https://github.com/whai362/PSENet/issues/15

"""
def detect(seg_maps, image_w, image_h, min_area_thresh=10, seg_map_thresh=0.5, ratio = 1):
    '''
    restore text boxes from score map and geo map
    :param seg_maps:
    :param timer:
    :param min_area_thresh:
    :param seg_map_thresh: threshhold for seg map
    :param ratio: compute each seg map thresh
    :return:
    '''
    if len(seg_maps.shape) == 4:
        seg_maps = seg_maps[-1, :, :]
    #get kernals, sequence: 0->n, max -> min
    kernals = []
    one = np.ones_like(seg_maps[-1], dtype=np.uint8)
    zero = np.zeros_like(seg_maps[-1], dtype=np.uint8)
    thresh = seg_map_thresh
    for i in range(seg_maps.shape[0]-1, -1, -1):
        if i not in [1, 2, 6]:
            continue
        
        kernal = np.where(seg_maps[i]>thresh, one, zero)
        # print("index:", i)
        # cv2.imshow('mask_res', kernal * 255)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        kernals.append(kernal)
        thresh = seg_map_thresh*ratio
    mask_res = pse_poster.pse(kernals, min_area_thresh)
    
    mask_res = np.array(mask_res)
    
    mask_res_resized = cv2.resize(mask_res, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
    boxes = []
    
    points = np.argwhere(mask_res_resized==1)
    points = points[:, (1,0)]
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    boxes.append(box)

    return np.array(boxes), kernals

def inference(data_root, ckpt, out_dir='result', target_size=1024, no_write_images=True, ctx=mx.cpu()):
    """
    Args:
        data_root: root dir of image
        out_dir: save output into out_dir
        ckpt: checkpoint for trained model
        target_size: resize img to target size to inference
    """
    # load weights
    net = PSENet(num_kernels=7, ctx=ctx)
    net.load_parameters(ckpt)
    
    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    imglst = glob.glob1(data_root, '*g')
    for imname in imglst:
        im = cv2.imread(os.path.join(data_root, imname))
        im_res, (ratio_h, ratio_w) = resize_image(im, target_size)
        h, w, _ = im_res.shape
        im_res = mx.nd.array(im_res)
        im_res = trans(im_res)
        # prediction kernel and segmentation result
        predict_kernels = net(im_res.expand_dims(axis=0))
        score_map = predict_kernels[0][6, :, :]
        score_map = mx.nd.where(score_map > 0.5, mx.nd.ones_like(score_map), mx.nd.zeros_like(score_map)) * 255
        # cv2.imshow('score_map', score_map.asnumpy())
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # post process
        start_time = time.time()
        
        # get result
        boxes, kernels = detect(predict_kernels[0].asnumpy(), w, h)
        
        # draw result on image and save result into txt
        
        if boxes is not None:
            boxes = boxes.reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h
            h, w, _ = im.shape
            boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, w)
            boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, h)

        duration = time.time() - start_time
        logging.info('[timing] {}, objects {}'.format(duration, len(boxes)))

        # save to file
        if boxes is not None:
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            res_filename = os.path.join(out_dir, ".".join(imname.split('.')[:-1])+'.txt')
            with open(res_filename, 'w') as f:
                num =0
                for i in xrange(len(boxes)):
                    # to avoid submitting errors
                    box = boxes[i]
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                        continue
                    num += 1
                    f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                        box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1]))
                    cv2.polylines(im, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=2)
        if not no_write_images:
            img_path = os.path.join(out_dir, os.path.basename(imname))
            cv2.imwrite(img_path, im)

if __name__ == "__main__":
    data_root = sys.argv[1]
    ckpt = sys.argv[2]
    if len(sys.argv) < 2:
        print("python eval.py data_root ckpt")
    inference(data_root, ckpt, no_write_images=False)






