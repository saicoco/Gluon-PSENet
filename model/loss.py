# coding=utf-8
from mxnet import gluon

class DiceLoss(gluon.loss.Loss):

    def __init__(self, cls_weight=0.01, iou_weight=1.0, angle_weight=20, weight=None, batch_axis=0, **kwargs):
        super(DiceLoss, self).__init__(weight=weight, batch_axis=batch_axis, **kwargs)
        self.cls_weight = cls_weight
        self.iou_weight = iou_weight
        self.angle_weight = angle_weight

    def hybrid_forward(self, F, score_gt, score_pred, training_masks, *args, **kwargs):

        # classification loss
        eps = 1e-5
        intersection = F.sum(score_gt * score_pred * training_masks)
        union = F.sum(training_masks * score_gt) + F.sum(training_masks * score_pred) + eps
        dice_loss = 1. - (2 * intersection / union)

        # TODO: 完成OHEM

        return dice_loss

