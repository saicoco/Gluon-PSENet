# coding=utf-8
from mxnet.gluon.data.dataset import Dataset
from mxnet.gluon.data.vision import transforms
from util import random_crop, random_rotate, random_scale, shrink_polys, parse_lines, save_images, rescale
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
        im, text_polys = rescale(im, text_polys)
        score_maps, training_mask = shrink_polys(im, polys=text_polys, tags=text_tags, mini_scale_ratio=0.5, num_kernels=6)
        imgs = [im] + score_maps + [training_mask]

        # random scale, random rotate, random crop
        imgs = random_scale(imgs)
        if np.random.uniform(-1, 1) > 0.6:
            imgs = random_rotate(imgs)
        imgs = random_crop(imgs, self.input_size)

        image, score_map, train_mask = imgs[0], imgs[1:-1], imgs[-2:-1]
        if self.debug:
            im_show = np.where(score_map[-1]==1, image[:,:,0], np.zeros_like(score_map[0]))
            cv2.imshow('img', im_show)
            cv2.waitKey()

        image, score_map, train_mask = mx.nd.array(image), mx.nd.array(score_map), mx.nd.array(train_mask)
        image = self.trans(image)
        return mx.nd.Concat(image, score_map, train_mask, dim=0)

    def __len__(self):
        return self.length

if __name__ == '__main__':
    from mxnet.gluon.data import dataloader
    import sys
    root_dir = sys.argv[1]
    icdar = ICDAR(data_dir=root_dir, debug=True)
    loader = dataloader.DataLoader(dataset=icdar, batch_size=10)
    for k, i in enumerate(loader):
        img = i[0, :, :, :].asnumpy()
        kernels = i[0, :, :, 3:6].asnumpy()
        print img.shape
        if k==10:
            # cv2.destroyAllWindows()``
            break


