# coding=utf-8
from mxnet.gluon import trainer
from datasets.dataloader import ICDAR
from mxnet.gluon.data import DataLoader
from model.net import PSENet
import mxnet as mx
from mxnet.gluon import Trainer
from model.loss import DiceLoss
from mxnet import autograd

def train(data_dir, pretrain_model, epoches=30, lr=0.001, batch_size=10, ctx=mx.cpu(), verbose_step=10):

    icdar_loader = ICDAR(data_dir=data_dir)
    loader = DataLoader(icdar_loader, batch_size=batch_size, shuffle=True)
    net = PSENet(num_kernels=6, ctx=ctx)
    # initial params
    net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=ctx)
    # net.load_parameters(pretrain_model, ctx=ctx, allow_missing=True, ignore_extra=True)
    pse_loss = DiceLoss(lam=0.7)
    trainer = Trainer(net.collect_params(), 'adam', {'learning_rate': lr})

    for e in range(epoches):
        cumulative_loss = 0

        for i, data in enumerate(loader):
            data = data.as_in_context(ctx).transpose((0, 3, 1, 2))
            im = data[:, :3, :, :]
            kernels = data[:, 3:9, ::4, ::4]
            training_masks = data[:, 9:, ::4, ::4]

            with autograd.record():
                kernels_pred = net(im)
                loss = pse_loss(kernels, kernels_pred, training_masks)
            loss.backward()
            trainer.step(batch_size)
            if i%verbose_step==0:
                print("step: {}, loss: {}".format(i * batch_size, mx.nd.mean(loss).asscalar()))
            cumulative_loss += mx.nd.mean(loss).asscalar()
        print("Epoch {}, loss: {}".format(e, cumulative_loss))

if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1]
    pretrain_model = sys.argv[2]
    if len(sys.argv) < 2:
        print("Usage: python train.py $data_dir $pretrain_model")
    train(data_dir=data_dir, pretrain_model=pretrain_model)

