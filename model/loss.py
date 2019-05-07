# coding=utf-8
from mxnet import gluon
from mxnet import nd as F

class DiceLoss_with_OHEM(gluon.loss.Loss):

    def __init__(self, lam=0.7, weight=None, batch_axis=0, debug=False, **kwargs):
        super(DiceLoss_with_OHEM, self).__init__(weight=weight, batch_axis=batch_axis, **kwargs)
        self.lam = lam
        self.kernel_loss = 0.
        self.C_loss = 0.

        self.debug = debug
    def _ohem_single(self, score_gt, score_pred, training_masks):
        if self.debug:
            print("score_gt_shape:", score_gt.shape, "score_pred_shape:", score_pred.shape, \
                "train_mask_shape:", training_masks.shape)
        pos_gt_thres = F.where(score_gt > 0.5, F.ones_like(score_gt), F.zeros_like(score_gt))
        pos_num = F.sum(pos_gt_thres) - F.sum(pos_gt_thres * training_masks)

        if pos_num == 0:
            selected_mask = training_masks
            return selected_mask
        
        neg_lt_thres = F.where(score_gt <= 0.5, F.ones_like(score_gt), F.zeros_like(score_gt))
        neg_num = F.sum(neg_lt_thres)
        neg_num = min(pos_num * 3, neg_num)

        if neg_num == 0:
            selected_mask = training_masks
            return training_masks
        neg_score = neg_lt_thres * score_pred
        neg_score_sorted = F.sort(neg_score.reshape(-1), is_ascend=0, axis=None)
        threshold = neg_score_sorted[neg_num - 1]
        score_gt_thres = F.where(score_pred >= threshold, F.ones_like(score_pred), F.zeros_like(score_pred))

        trained_sample_mask = F.logical_or(score_gt_thres, pos_gt_thres)
        selected_mask = F.logical_and(trained_sample_mask, training_masks)

        return selected_mask

        
    def hybrid_forward(self, F, score_gt, kernel_gt, score_pred, training_masks, *args, **kwargs):
        
        # cal ohem mask
        selected_masks = []
        for i in range(score_gt.shape[0]):
            # cal for text region
            selected_mask = self._ohem_single(score_gt[i:i+1], score_pred[i:i+1], training_masks[i:i+1])
            selected_masks.append(selected_mask)
        selected_masks = F.concat(*selected_masks, dim=0)

        s1, s2, s3, s4, s5, s6 = F.split(kernel_gt, num_outputs=6, axis=1)
        s1_pred, s2_pred, s3_pred, s4_pred, s5_pred, s6_pred, C_pred = F.split(score_pred, num_outputs=7, axis=1)

        # for text map
        eps = 1e-5
        intersection = F.sum(score_gt * C_pred * selected_masks)
        union = F.sum(score_gt * score_gt * selected_masks) + F.sum(C_pred * C_pred * selected_mask) + eps
        C_dice_loss = 1. - F.mean((2 * intersection / union))


        # loss for kernel
        kernel_dices = []
        for s, s_pred in zip([s1, s2, s3, s4, s5, s6], [s1_pred, s2_pred, s3_pred, s4_pred, s5_pred, s6_pred]):
            kernel_mask = F.where(s > 0.5, F.ones_like(s), F.zeros_like(s))
            kernel_intersection = F.sum(s * s_pred * training_masks * kernel_mask)
            kernel_union = F.sum(training_masks * s * s * kernel_mask) + F.sum(
                training_masks * s_pred * s_pred * kernel_mask) + eps
            kernel_dice = 2. * kernel_intersection / kernel_union
            kernel_dices.append(kernel_dice.asscalar())
        kernel_dice_loss =1. - F.mean(F.array(kernel_dices))

        self.kernel_loss = kernel_dice_loss
        self.C_loss = C_dice_loss

        loss = self.lam * C_dice_loss + (1. - self.lam) * kernel_dice_loss
        
        return loss


class DiceLoss(gluon.loss.Loss):

    def __init__(self, lam=0.7, weight=None, batch_axis=0, **kwargs):
        super(DiceLoss, self).__init__(weight=weight, batch_axis=batch_axis, **kwargs)
        self.lam = lam
        self.kernel_loss = 0.
        self.C_loss = 0.

    def hybrid_forward(self, F, score_gt, kernel_gt, score_pred, training_masks, *args, **kwargs):
        s1, s2, s3, s4, s5, s6 = F.split(kernel_gt, num_outputs=6, axis=1)
        s1_pred, s2_pred, s3_pred, s4_pred, s5_pred, s6_pred, C_pred = F.split(score_pred, num_outputs=7, axis=1)

        # classification loss
        eps = 1e-5
        intersection = F.sum(score_gt * C_pred * training_masks)
        union = F.sum(training_masks * score_gt * score_gt) + F.sum(training_masks * C_pred * C_pred) + eps
        C_dice_loss = 1. - F.mean((2 * intersection / union))
        # print("C_dice_loss:", C_dice_loss)
        # loss for kernel
        kernel_dices = []
        for s, s_pred in zip([s1, s2, s3, s4, s5, s6], [s1_pred, s2_pred, s3_pred, s4_pred, s5_pred, s6_pred]):
            kernel_mask = F.where((score_gt * training_masks > 0.5), F.ones_like(s), F.zeros_like(s))
            kernel_intersection = F.sum(s * s_pred * training_masks * kernel_mask)
            kernel_union = F.sum(training_masks * s * s * kernel_mask) + F.sum(
                training_masks * s_pred * s_pred * kernel_mask) + eps
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
    np.random.seed(29999)
    loss = DiceLoss_with_OHEM(lam=0.7, debug=True)
    # loss = DiceLoss()
    for i in range(1):
        score_gt = F.array(np.random.normal(size=(6, 1, 128, 128)))
        x = F.array(np.random.normal(size=(6, 6, 128, 128)))
        x.attach_grad()
        x_pred = F.array(np.random.normal(size=(6, 7, 128, 128)))
        mask = F.ones(shape=(6, 1, 128, 128))
        with autograd.record():
            tmp_loss = loss.forward(score_gt, x, x_pred, mask)
            # tmp_loss.backward()
        print tmp_loss


