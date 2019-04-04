# coding=utf-8
from mxnet.gluon import HybridBlock, nn
import mxnet as mx
from gluoncv import model_zoo
from gluoncv.model_zoo.resnetv1b import resnet50_v1b
from feature import FPNFeatureExpander

# class PSENet_old(HybridBlock):

#     def __init__(self, num_kernels, ctx=mx.cpu(), pretrained=True):
#         super(PSENet_old, self).__init__()
#         self.num_kernels = num_kernels

#         pretrained = model_zoo.get_model(name='resnet50_v1b', pretrained=pretrained, ctx=ctx)
#         self.conv1 = pretrained.conv1
#         self.bn1 = pretrained.bn1
#         self.relu = pretrained.relu
#         self.maxpool = pretrained.maxpool
#         self.layer1 = pretrained.layer1
#         self.layer2 = pretrained.layer2
#         self.layer3 = pretrained.layer3
#         self.layer4 = pretrained.layer4

#         self.CONV1 = nn.HybridSequential(prefix='conv1')
#         with self.CONV1.name_scope():
#             self.CONV1.add(self.conv1)
#             self.CONV1.add(self.bn1)
#             self.CONV1.add(self.relu)
#             self.CONV1.add(self.maxpool)

#         # decoder
#         self.encoder_layer4 = nn.Conv2D(channels=256, kernel_size=(3, 3), padding=(1, 1), activation='relu')

#         self.decoder_conv1 = nn.HybridSequential(prefix='decoder_conv1')
#         with self.decoder_conv1.name_scope():
#             self.dconv1_1x1 = nn.Conv2D(channels=256, kernel_size=(1, 1))
#             self.decoder_conv1.add(self.dconv1_1x1)
#             self.decoder_conv1.add(nn.BatchNorm())
#             self.decoder_conv1.add(nn.Activation('relu'))

#         self.encoder_layer3 = nn.Conv2D(channels=256, kernel_size=(3, 3), padding=(1, 1), activation='relu')

#         self.decoder_conv2 = nn.HybridSequential(prefix='decoder_conv2')
#         with self.decoder_conv2.name_scope():
#             self.dconv2_1x1 = nn.Conv2D(channels=256, kernel_size=(1, 1))
#             self.decoder_conv2.add(self.dconv2_1x1)
#             self.decoder_conv2.add(nn.BatchNorm())
#             self.decoder_conv2.add(nn.Activation('relu'))
            
#         self.encoder_layer2 = nn.Conv2D(channels=256, kernel_size=(3, 3), padding=(1, 1), activation='relu')

#         self.decoder_conv3 = nn.HybridSequential(prefix='decoder_conv3')
#         with self.decoder_conv3.name_scope():
#             self.dconv3_1x1 = nn.Conv2D(channels=256, kernel_size=(1, 1))
#             self.decoder_conv3.add(self.dconv3_1x1)
#             self.decoder_conv3.add(nn.BatchNorm())
#             self.decoder_conv3.add(nn.Activation('relu'))
            
#         self.encoder_layer1 = nn.Conv2D(channels=256, kernel_size=(3, 3), padding=(1, 1), activation='relu')

#         self.reduce_conv = nn.HybridSequential(prefix='reduce_conv')
#         with self.reduce_conv.name_scope():
#             self.conv1_reduce = nn.Conv2D(channels=256, kernel_size=(1, 1))
#             self.reduce_conv.add(self.conv1_reduce)
#             self.reduce_conv.add(nn.BatchNorm())
#             self.reduce_conv.add(nn.Activation('relu'))
            
#         self.decoder_out = nn.HybridSequential(prefix='decoder_out')
#         with self.decoder_out.name_scope():
#             self.conv1_out = nn.Conv2D(channels=self.num_kernels, kernel_size=(1, 1), activation='sigmoid')
#             self.decoder_out.add(self.conv1_out)
            
#     def hybrid_forward(self, F, x, *args, **kwargs):
#         # forward
#         N, C, H, W =  x.shape
#         conv1 = self.CONV1(x) # stride = 4
#         layer1 = self.layer1(conv1)
#         layer2 = self.layer2(layer1) # stride = 8
#         layer3 = self.layer3(layer2) # stride = 16
#         layer4 = self.layer4(layer3) # struide = 32

#         # decoder: add + conv1x1 + conv3x3

#         upool_layer4 = F.contrib.BilinearResize2D(layer4, height=H//16, width=W//16, name='unpool1')
#         decode_layer4 = self.decoder_conv1(upool_layer4) + self.encoder_layer3(layer3)

#         upool_layer3 = F.contrib.BilinearResize2D(decode_layer4, height=H // 8, width=W // 8, name='unpool2')
#         decode_layer3 = self.decoder_conv2(upool_layer3) + self.encoder_layer2(layer2)

#         upool_layer2 = F.contrib.BilinearResize2D(decode_layer3, height=H // 4, width=W // 4, name='unpool3')
#         decode_layer2 = self.decoder_conv3(upool_layer2) + self.encoder_layer1(conv1)

#         # feature concanatetion
#         feature_layer4 = F.contrib.BilinearResize2D(self.encoder_layer4(layer4), height=H//4, width=W//4, name='feature_upsample_1')
#         feature_layer3 = F.contrib.BilinearResize2D(decode_layer4, height=H // 4, width=W // 4, name='feature_upsample_2')
#         feature_layer2 = F.contrib.BilinearResize2D(decode_layer3, height=H // 4, width=W // 4, name='feature_upsample_3')
#         feature_layer1 = F.contrib.BilinearResize2D(decode_layer2, height=H // 4, width=W // 4, name='feature_upsample_4')
#         feature_maps = F.concat(feature_layer1, feature_layer2, feature_layer3, feature_layer4, dim=1)

#         feature_reduce = self.reduce_conv(feature_maps)
#         feature_out = self.decoder_out(feature_reduce)
#         return feature_out


class PSENet(HybridBlock):

    def __init__(self, num_kernels, scale=1.0, ctx=mx.cpu(), pretrained=True,  **kwargs):
        super(PSENet, self).__init__()
        self.num_kernels = num_kernels
        base_network = resnet50_v1b(pretrained=pretrained, dilated=False,
                                use_global_stats=True, **kwargs)
        self.features = FPNFeatureExpander(
        network=base_network,
        outputs=['layers1_relu8_fwd', 'layers2_relu11_fwd', 'layers3_relu17_fwd',
                 'layers4_relu8_fwd'], num_filters=[256, 256, 256, 256], use_1x1=True,
        use_upsample=True, use_elewadd=True, use_p6=False, no_bias=False, pretrained=pretrained)
        self.scale = scale
        self.extrac_convs = []
        weight_init = mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2.)
        for i in range(4):
            extra_conv = nn.HybridSequential(prefix='extra_conv_{}'.format(i))
            with extra_conv.name_scope():
                extra_conv.add(nn.Conv2D(256, 3, 1, 1))
                extra_conv.add(nn.BatchNorm())
                extra_conv.add(nn.Activation('relu'))
            extra_conv.initialize(weight_init)
            self.extrac_convs.append(extra_conv)
        
        self.decoder_out = nn.HybridSequential(prefix='decoder_out')
        with self.decoder_out.name_scope():
            self.decoder_out.add(nn.Conv2D(256, 3, 1, 1))
            self.decoder_out.add(nn.BatchNorm())
            self.decoder_out.add(nn.Activation('relu'))
            self.decoder_out.add(nn.Conv2D(self.num_kernels, 1, 1))
            self.decoder_out.initialize(weight_init)

    def hybrid_forward(self, F, x, **kwargs):
        # output: c4 -> c1 [1/4, 1/8, 1/16. 1/32]
        fpn_features = self.features(x)
        concat_features = []
        scales = [1, 2, 4, 8]
        for i, C in enumerate(fpn_features):
            extrac_C = self.extrac_convs[i](C)
            print("{}: shape:{}".format(i, extrac_C.shape))
            up_C = F.UpSampling(extrac_C, scale=scales[i], sample_type='nearest', name="extra_upsample_{}".format(i))
            concat_features.append(up_C)
        concat_output = F.concat(*concat_features, dim=1)
        output = self.decoder_out(concat_output)
        print(output.shape)
        if self.scale >= 1.0:
            output = F.UpSampling(output, scale=4, sample_type='nearest', name="final_upsampling")
        return output

if __name__ == '__main__':
    import numpy as np
    fpn = PSENet(num_kernels=6, pretrained=True)
    fpn.initialize(ctx=mx.cpu())
    x = mx.nd.array([np.random.normal(size=(3, 512, 512))])
    print map(lambda x:x.shape, fpn(x))


