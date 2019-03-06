# coding=utf-8
from mxnet.gluon import HybridBlock, nn
import mxnet as mx
from gluoncv import model_zoo

class PSENet(HybridBlock):

    def __init__(self, num_kernels, channels=None, ctx=mx.cpu(), pretrained=True):
        super(PSENet, self).__init__()
        self.num_kernels = num_kernels

        pretrained = model_zoo.get_model(name='resnet50_v1s', pretrained=pretrained, ctx=ctx)
        self.conv1 = pretrained.conv1
        self.bn1 = pretrained.bn1
        self.relu = pretrained.relu
        self.maxpool = pretrained.maxpool
        self.layer1 = pretrained.layer1
        self.layer2 = pretrained.layer2
        self.layer3 = pretrained.layer3
        self.layer4 = pretrained.layer4

        self.CONV1 = nn.HybridSequential(prefix='conv1')
        with self.CONV1.name_scope():
            self.CONV1.add(self.conv1)
            self.CONV1.add(self.bn1)
            self.CONV1.add(self.relu)
            self.CONV1.add(self.maxpool)

        # decoder
        self.encoder_layer4 = nn.Conv2D(channels=256, kernel_size=(1, 1), activation='relu')

        self.decoder_conv1 = nn.HybridSequential(prefix='decoder_conv1')
        with self.decoder_conv1.name_scope():
            self.dconv1_3x3 = nn.Conv2D(channels=256, kernel_size=(3, 3), padding=(1, 1), activation='relu')
            self.dconv1_1x1 = nn.Conv2D(channels=256, kernel_size=(1, 1), activation='relu')
            self.decoder_conv1.add(self.dconv1_3x3)
            self.decoder_conv1.add(self.dconv1_1x1)
        self.encoder_layer3 = nn.Conv2D(channels=256, kernel_size=(1, 1), activation='relu')

        self.decoder_conv2 = nn.HybridSequential(prefix='decoder_conv2')
        with self.decoder_conv2.name_scope():
            self.dconv2_3x3 = nn.Conv2D(channels=128, kernel_size=(3, 3), padding=(1, 1), activation='relu')
            self.dconv2_1x1 = nn.Conv2D(channels=128, kernel_size=(1, 1), activation='relu')
            self.decoder_conv2.add(self.dconv2_3x3)
            self.decoder_conv2.add(self.dconv2_1x1)
        self.encoder_layer2 = nn.Conv2D(channels=128, kernel_size=(1, 1), activation='relu')

        self.decoder_conv3 = nn.HybridSequential(prefix='decoder_conv3')
        with self.decoder_conv3.name_scope():
            self.dconv3_3x3 = nn.Conv2D(channels=64, kernel_size=(3, 3), padding=(1, 1), activation='relu')
            self.dconv3_1x1 = nn.Conv2D(channels=64, kernel_size=(1, 1), activation='relu')
            self.decoder_conv3.add(self.dconv3_3x3)
            self.decoder_conv3.add(self.dconv3_1x1)
        self.encoder_layer1 = nn.Conv2D(channels=64, kernel_size=(1, 1), activation='relu')

        self.reduce_conv = nn.HybridSequential(prefix='reduce_conv')
        with self.reduce_conv.name_scope():
            self.conv1_reduce = nn.Conv2D(channels=256, kernel_size=(3, 3), padding=(1, 1), activation='relu')
            self.reduce_conv.add(self.conv1_reduce)
        self.decoder_out = nn.HybridSequential(prefix='decoder_out')
        with self.decoder_out.name_scope():
            self.conv1_out = nn.Conv2D(channels=self.num_kernels, kernel_size=(1, 1), activation='sigmoid')
            self.decoder_out.add(self.conv1_out)
    def hybrid_forward(self, F, x, *args, **kwargs):
        # forward
        N, C, H, W =  x.shape
        conv1 = self.CONV1(x) # stride = 4
        layer1 = self.layer1(conv1)
        layer2 = self.layer2(layer1) # stride = 8
        layer3 = self.layer3(layer2) # stride = 16
        layer4 = self.layer4(layer3) # struide = 32

        # decoder: add + conv1x1 + conv3x3

        upool_layer4 = F.contrib.BilinearResize2D(layer4, height=H//16, width=W//16, name='unpool1')
        decode_layer4 = self.decoder_conv1(upool_layer4) + self.encoder_layer3(layer3)

        upool_layer3 = F.contrib.BilinearResize2D(decode_layer4, height=H // 8, width=W // 8, name='unpool2')
        decode_layer3 = self.decoder_conv2(upool_layer3) + self.encoder_layer2(layer2)

        upool_layer2 = F.contrib.BilinearResize2D(decode_layer3, height=H // 4, width=W // 4, name='unpool3')
        decode_layer2 = self.decoder_conv3(upool_layer2) + self.encoder_layer1(conv1)

        # feature concanatetion
        feature_layer4 = F.contrib.BilinearResize2D(self.encoder_layer4(layer4), height=H//4, width=W//4, name='feature_upsample_1')
        feature_layer3 = F.contrib.BilinearResize2D(decode_layer4, height=H // 4, width=W // 4, name='feature_upsample_2')
        feature_layer2 = F.contrib.BilinearResize2D(decode_layer3, height=H // 4, width=W // 4, name='feature_upsample_3')
        feature_layer1 = F.contrib.BilinearResize2D(decode_layer2, height=H // 4, width=W // 4, name='feature_upsample_4')
        feature_maps = F.concat(feature_layer1, feature_layer2, feature_layer3, feature_layer4, dim=1)

        feature_reduce = self.reduce_conv(feature_maps)
        feature_out = self.decoder_out(feature_reduce)
        return feature_out



if __name__ == '__main__':
    import numpy as np
    fpn = PSENet(num_kernels=6, pretrained=False)
    fpn.initialize(ctx=mx.cpu())
    x = mx.nd.array([np.random.normal(size=(3, 512, 512))])

    print map(lambda x:x.shape, fpn(x))


