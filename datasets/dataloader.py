from mxnet.gluon.data import dataset
from mxnet.gluon.data.vision import transforms
from util import random_crop, random_rotate, random_scale, shrink_polys, parse_lines
import os
import glob
import cv2

class ICDAR(dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.imglst = glob.glob1(self.data_dir, '*g')
        self.length = len(self.imglst)

    def __getitem__(self, item):
        img_name = self.imglst[item]
        prefix = ".".join(img_name.split('.')[:-1])
        label_name = prefix + '.txt'
        text_polys = parse_lines(os.path.join(self.data_dir, label_name))
        im = cv2.imread(os.path.join(self.data_dir, img_name))
        score_maps = shrink_polys(im, polys=text_polys, mini_scale_ratio=0.5, num_kernels=6)
        imgs = [im] + score_maps

        # random scale, random rotate, random crop
        imgs = random_scale(imgs)
        imgs = random_rotate(imgs)
        imgs = random_crop(imgs, (640, 640))
        return imgs

    def __len__(self):
        return self.length

