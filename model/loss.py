# coding=utf-8
from mxnet import gluon
from mxnet import nd as F

class DiceLoss_with_OHEM(gluon.loss.Loss):

    def __init__(self, lam=0.7, weight=None, batch_axis=0, **kwargs):
        super(DiceLoss, self).__init__(weight=weight, batch_axis=batch_axis, **kwargs)
        self.lam = lam

    def hybrid_forward(self, F, score_gt, score_pred, training_masks, *args, **kwargs):

        s1, s2, s3, s4, s5, C = F.split(score_gt, num_outputs=6, axis=1)
        s1_pred, s2_pred, s3_pred, s4_pred, s5_pred, C_pred = F.split(score_pred, num_outputs=6, axis=1)

        all_pos_samples = F.sum(C)
        all_neg_samples = F.sum(F.ones_like(C) - C)
        all_samples = all_neg_samples + all_pos_samples
        all_neg_samples = 3 * all_pos_samples if (3 * all_pos_samples) < all_neg_samples else all_neg_samples

        # get negative sample and positive for C map
        negative_sig_out = (F.ones_like(C) - C) * C_pred * training_masks

        C_topk_neg_mask = F.topk(negative_sig_out.reshape((1, -1)), ret_typ='mask', k=int(all_neg_samples.asscalar())).reshape(negative_sig_out.shape)

        C_topk_mask = C_topk_neg_mask + C
        # classification loss
        eps = 1e-5
        intersection = F.sum(score_gt * score_pred * training_masks * C_topk_mask)
        union = F.sum(training_masks * score_gt * C_topk_mask) + F.sum(training_masks * score_pred * C_topk_mask) + eps
        C_dice_loss = 1. - (2 * intersection / union)


        # loss for kernel
        kernel_dices = []
        for s, s_pred in zip([s1, s2, s3, s4, s5], [s1_pred, s2_pred, s3_pred, s4_pred, s5_pred]):
            kernel_mask = F.where(s > 0.5, F.ones_like(s), F.zeros_like(s))
            kernel_intersection = F.sum(s * s_pred * training_masks * kernel_mask)
            kernel_union = F.sum(training_masks * s * kernel_mask) + F.sum(
                training_masks * s_pred * kernel_mask) + eps
            kernel_dice = 2. * kernel_intersection / kernel_union
            kernel_dices.append(kernel_dice.asscalar())
        kernel_dice_loss =1. - F.mean(F.array(kernel_dices))

        loss = self.lam * C_dice_loss + (1. - self.lam) * kernel_dice_loss
        
        return loss


class DiceLoss(gluon.loss.Loss):

    def __init__(self, lam=0.7, weight=None, batch_axis=0, **kwargs):
        super(DiceLoss, self).__init__(weight=weight, batch_axis=batch_axis, **kwargs)
        self.lam = lam
        self.kernel_loss = 0.
        self.C_loss = 0.

    def hybrid_forward(self, F, score_gt, score_pred, training_masks, *args, **kwargs):
        s1, s2, s3, s4, s5, C = F.split(score_gt, num_outputs=6, axis=1)
        s1_pred, s2_pred, s3_pred, s4_pred, s5_pred, C_pred = F.split(score_pred, num_outputs=6, axis=1)

        # classification loss
        eps = 1e-5
        intersection = F.sum(score_gt * score_pred * training_masks)
        union = F.sum(training_masks * score_gt) + F.sum(training_masks * score_pred) + eps
        C_dice_loss = 1. - (2 * intersection / union)
        # print("C_dice_loss:", C_dice_loss)
        # loss for kernel
        kernel_dices = []
        for s, s_pred in zip([s1, s2, s3, s4, s5], [s1_pred, s2_pred, s3_pred, s4_pred, s5_pred]):
            kernel_mask = F.where(s > 0.5, F.ones_like(s), F.zeros_like(s))
            kernel_intersection = F.sum(s * s_pred * training_masks * kernel_mask)
            kernel_union = F.sum(training_masks * s * kernel_mask) + F.sum(
                training_masks * s_pred * kernel_mask) + eps
            kernel_dice = 2. * kernel_intersection / kernel_union
            kernel_dices.append(kernel_dice.asscalar())
        kernel_dice_loss = 1. - F.mean(F.array(kernel_dices))
        # print("kernel_loss:", kernel_dice_loss)
        self.C_loss = C_dice_loss
        self.kernel_loss = kernel_dice_loss
        
        loss = self.lam * C_dice_loss + (1. - self.lam) * kernel_dice_loss

        return loss



if __name__ == '__main__':
    import numpy as np
    from mxnet import autograd
    loss = DiceLoss()
    for i in range(2):
        x = F.array(np.random.normal(size=(1, 6, 128, 128)))
        x.attach_grad()
        x_pred = F.array(np.random.normal(size=(1, 6, 128, 128)))
        mask = F.ones(shape=(1, 1, 128, 128))
        with autograd.record():
            tmp_loss = loss.forward(x, x_pred, mask)
        tmp_loss.backward()
        print x.grad


