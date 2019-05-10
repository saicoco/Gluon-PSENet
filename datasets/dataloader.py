# coding=utf-8
from mxnet.gluon.data.dataset import Dataset
from mxnet.gluon.data.vision import transforms
from util import random_crop, random_rotate, random_scale, shrink_polys, parse_lines, save_images, random_horizontal_flip
import os
import glob
import cv2
from PIL import Image
import mxnet as mx
import numpy as np
from mxnet.gluon.data.vision import transforms

class ICDAR(Dataset):
    def __init__(self, data_dir, strides=4, input_size=(640, 640), debug=False):
        super(ICDAR, self).__init__()
        self.data_dir = data_dir
        self.imglst = glob.glob1(self.data_dir, '*g')
        self.length = len(self.imglst)
        self.input_size = input_size
        self.strides = strides
        self.debug = debug
        self.trans = transforms.Compose([
            transforms.RandomColorJitter(brightness = 32.0 / 255, saturation = 0.5),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])


    def __getitem__(self, item):
        img_name = self.imglst[item]
        prefix = ".".join(img_name.split('.')[:-1])
        label_name = prefix + '.txt'
        text_polys, text_tags = parse_lines(os.path.join(self.data_dir, label_name))
        # im = cv2.imread(os.path.join(self.data_dir, img_name))
        im = Image.open(os.path.join(self.data_dir, img_name)).convert('RGB')
        im = np.array(im)[:, :, :3]
        im, text_polys = random_scale(im, text_polys)
        score_maps, kernel_maps, training_mask = shrink_polys(im, polys=text_polys, tags=text_tags, mini_scale_ratio=0.5, num_kernels=6)
        imgs = [im, score_maps, kernel_maps, training_mask]

        # random_flip,random rotate, random crop

        imgs = random_horizontal_flip(imgs)
        # imgs = random_rotate(imgs)
        imgs = random_crop(imgs, self.input_size)

        image, score_map, kernel_map, training_mask = imgs[0], imgs[1], imgs[2], imgs[3]
        if self.debug:
            im_show = np.concatenate([score_map, kernel_map[:, :, 0], kernel_map[:, :, 5]], axis=1)
            cv2.imshow('img', image)
            cv2.imshow('score_map', im_show)
            cv2.waitKey()
        image = mx.nd.array(image)
        score_map = mx.nd.array(score_map, dtype=np.float32)
        kernal_map = mx.nd.array(kernel_map, dtype=np.float32)
        training_mask = mx.nd.array(training_mask, dtype=np.float32)
        trans_image = self.trans(image)
        return trans_image, score_map, kernel_map, training_mask, transforms.ToTensor()(image)

    def __len__(self):
        return self.length

if __name__ == '__main__':
    from mxnet.gluon.data import dataloader
    import sys
    
    root_dir = sys.argv[1]
    icdar = ICDAR(data_dir=root_dir, debug=True)
    loader = dataloader.DataLoader(dataset=icdar, batch_size=1)
    for k, item in enumerate(loader):
        img, score, kernel, training_mask, ori_img = item
        img = img.asnumpy()
        kernels = kernel.asnumpy()
        print img.shape, score
        if k==10:
            break


