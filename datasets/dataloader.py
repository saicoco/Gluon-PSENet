# coding=utf-8
from mxnet.gluon.data.dataset import Dataset
from mxnet.gluon.data.vision import transforms
from util import random_crop, random_rotate, random_scale, shrink_polys, parse_lines, save_images, rescale
import os
import glob
import cv2
import mxnet as mx

class ICDAR(Dataset):
    def __init__(self, data_dir, input_size=(640, 640)):
        super(ICDAR, self).__init__()
        self.data_dir = data_dir
        self.imglst = glob.glob1(self.data_dir, '*g')
        self.length = len(self.imglst)
        self.input_size = input_size

    def __getitem__(self, item):
        img_name = self.imglst[item]
        prefix = ".".join(img_name.split('.')[:-1])
        label_name = prefix + '.txt'
        text_polys, text_tags = parse_lines(os.path.join(self.data_dir, label_name))
        im = cv2.imread(os.path.join(self.data_dir, img_name))
        im, text_polys = rescale(im, text_polys)
        score_maps, training_mask = shrink_polys(im, polys=text_polys, tags=text_tags, mini_scale_ratio=0.5, num_kernels=6)
        imgs = [im] + score_maps + [training_mask]

        # random scale, random rotate, random crop
        imgs = random_rotate(imgs)
        imgs = random_scale(imgs)
        imgs = random_crop(imgs, self.input_size)
        image, score_map, train_mask = imgs[0], imgs[1:-1], imgs[-1]
        return mx.nd.array(image), mx.nd.array(score_map), mx.nd.array(train_mask)

    def __len__(self):
        return self.length

if __name__ == '__main__':
    from mxnet.gluon.data import dataloader
    import sys
    root_dir = sys.argv[1]
    icdar = ICDAR(data_dir=root_dir)
    loader = dataloader.DataLoader(dataset=icdar, batch_size=2)
    for k, i in enumerate(loader):
        print i
        if k==3:
            break


