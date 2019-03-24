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

def resize(im, target_size):
    return im

def get_result(res_map):
    """
    process res_map to contour points
    """
    return 

def draw_pts_on_img(img, pts):
    """
    draw pts on img
    """
    return img
def inference(data_root, out_dir, ckpt, target_size, ctx=mx.cpu()):
    """
    Args:
        data_root: root dir of image
        out_dir: save output into out_dir
        ckpt: checkpoint for trained model
        target_size: resize img to target size to inference
    """
    # load weights
    net = PSENet(num_kernels=6, ctx=ctx)
    net.load_parameters(ckpt)
    
    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    imglst = glob.glob1(data_root, '*g')
    for imname in imglst:
        im = cv2.imread(os.path.join(data_root, imname))
        im = resize(im, target_size)
        im = trans(im)
        # prediction kernel and segmentation result
        predict_kernels = net(im)
        # post process
        pred_map = pse_poster.pse(predict_kernels, 8)
        # get result
        points = get_result(pred_map)
        # draw result on image and save result into txt
        res_filename = os.path.join(out_dir, ".".join(imname.split('.')[:-1])+'.txt')

        # TODO: 完成后处理部分的画图、保存结果以及调试

    

    

    

