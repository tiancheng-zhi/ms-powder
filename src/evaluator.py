import numpy as np
import torch
import torch.nn.functional as F
import cv2


class Evaluator:

    def __init__(self, n_classes, bg_err):
        self.n_classes = n_classes
        self.bg_err = bg_err

        self.bg_confs = []
        self.pd_preds = []
        self.gt_nums = [[] for c in range(n_classes)]
        self.cls_bg_confs = [[] for c in range(n_classes)]

        self.tp = np.zeros(n_classes)
        self.fp = np.zeros(n_classes)
        self.num = np.zeros(n_classes)
        self.itp = np.zeros(n_classes)
        self.inum = np.zeros(n_classes)

        self.preds = []

    def register(self, label, prob):
        """
            label: H x W, n_classes-1 is bg
            prob: C x H x W
        """

        bg_conf = prob[-1,:,:]
        pd_pred = np.argmax(prob[:-1,:,:], axis=0)
        self.bg_confs.append(bg_conf)
        self.pd_preds.append(pd_pred)
        pred = np.argmax(prob, axis=0)
        self.preds.append(pred)
        for c in range(self.n_classes):
            gt_mask = (label == c)
            # for Powder Accuracy
            if gt_mask.any():
                if c < self.n_classes - 1:
                    pred_mask = (pd_pred == c)
                    self.gt_nums[c].append(gt_mask.sum())
                    self.cls_bg_confs[c].append(bg_conf[gt_mask * pred_mask])
                else:
                    self.cls_bg_confs[c] += list(bg_conf[gt_mask])
            # for IoU
                self.num[c] += gt_mask.sum()
                self.tp[c] += ((pred == c) * gt_mask).sum()
                self.inum[c] += 1
                self.itp[c] += ((pred == c) * gt_mask).sum() / gt_mask.sum()
            self.fp[c] += ((pred == c) * (1 - gt_mask)).sum()

    def evaluate(self):
        self.bg_conf_threshold = np.percentile(self.cls_bg_confs[-1], self.bg_err)
        accs = []
        for c in range(self.n_classes - 1):
            for i, cls_bg_conf in enumerate(self.cls_bg_confs[c]):
                acc = (cls_bg_conf < self.bg_conf_threshold).sum() / self.gt_nums[c][i]
                accs.append(acc)
        msa = np.mean(np.array(accs))

        self.bg_confs = np.array(self.bg_confs)
        self.pd_preds = np.array(self.pd_preds)
        predictions = self.pd_preds.copy()
        predictions[self.bg_confs >= self.bg_conf_threshold] = self.n_classes - 1

        iou = self.tp / (self.num + self.fp)
        miou = iou.mean()
        iiou = (self.itp * self.num / self.inum) / (self.num + self.fp)
        miiou = iiou.mean()
        return msa, predictions, miou, miiou, self.preds

