import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import RealDataset
from model import PowderNet
from utils import to_image, colormap, errormap
from evaluator import Evaluator
import skimage.io as io
import cv2
import collections


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--real-path', type=str, default='../real')
    parser.add_argument('--out-path', type=str, default='./result')
    parser.add_argument('--bg-err', type=float, default=1.0)
    parser.add_argument('--sdims', type=int, default=3)
    parser.add_argument('--schan', type=int, default=3)
    parser.add_argument('--compat', type=int, default=3)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    opt = parser.parse_args()
    return opt


def crf(prob, im, sdims, schan, compat, iters):
    if opt.channels == 965:
        bilateral_ch = [0,1,2,3,4,159,324,469,644,779,964]
    elif opt.channels == 4:
        bilateral_ch = [0,1,2,3]
    elif opt.channels == -961:
        bilateral_ch = [0,155,320,465,640,775,960]
    else:
        bilateral_ch = range(opt.n_channels)
    C, H, W = prob.shape
    U = unary_from_softmax(prob)
    d = dcrf.DenseCRF2D(H, W, C)
    d.setUnaryEnergy(U)
    pairwise_energy = create_pairwise_bilateral(sdims=(sdims, sdims), schan=(schan,), img=im[bilateral_ch, :, :], chdim=0)
    d.addPairwiseEnergy(pairwise_energy, compat=compat)
    Q_unary = d.inference(iters)
    Q_unary = np.array(Q_unary).reshape(-1, H, W)
    return Q_unary


def test(opt, test_loader, net, split):
    start_time = time.time()
    eva = Evaluator(opt.n_classes, opt.bg_err)
    eva_crf = Evaluator(opt.n_classes, opt.bg_err)
    ims = []
    labels = []

    net = net.eval()

    for iteration, batch in enumerate(test_loader):
        im, label = batch
        im = im.cuda()
        label = label.cuda()
        out = net(im)
        prob = F.softmax(out, dim=1)
        for i in range(opt.batch_size):
            prob_np = prob[i].detach().cpu().numpy()
            label_np = label[i].cpu().numpy()
            im_np = im[i].cpu().numpy()
            ims.append(to_image(im[i,:3,:,:]))
            labels.append(label_np)
            eva.register(label_np, prob_np)
            prob_crf = crf(prob_np, im_np, opt.sdims, opt.schan, opt.compat, opt.iters)
            eva_crf.register(label_np, prob_crf)
        print(str(iteration * opt.batch_size + i).zfill(2), time.time() - start_time, 'seconds')

    msa, preds_msa, miou, miiou, preds_miou = eva.evaluate()
    msa_crf, preds_msa_crf, miou_crf, miiou_crf, preds_miou_crf = eva_crf.evaluate()
    print('Pre-CRF:  MSA: {}   mIoU: {}   miIoU: {}'.format(round(msa * 100, 1), round(miou * 100, 1), round(miiou * 100, 1)))
    print('Post-CRF: MSA: {}   mIoU: {}   miIoU: {}'.format(round(msa_crf * 100, 1), round(miou_crf * 100, 1), round(miiou_crf * 100, 1)))
    for i, label in enumerate(labels):
        pred_msa = preds_msa[i]
        pred_msa_crf = preds_msa_crf[i]
        pred_miou = preds_miou[i]
        pred_miou_crf = preds_miou_crf[i]
        vis_im = ims[i]
        vis_label = colormap(label)
        vis_pred_msa = colormap(pred_msa)
        vis_pred_msa_crf = colormap(pred_msa_crf)
        vis_pred_miou = colormap(pred_miou)
        vis_pred_miou_crf = colormap(pred_miou_crf)
        vis_all = np.concatenate((
                      np.concatenate((vis_im, vis_label), axis=2),
                      np.concatenate((vis_pred_miou, vis_pred_miou_crf), axis=2)), axis=1)
        vis_all = vis_all.transpose((1, 2, 0))
        io.imsave(Path(opt.out_path) / split / (str(i).zfill(2) + '.png'), vis_all)
    return msa, miou, miiou, msa_crf, miou_crf, miiou_crf

if __name__ == '__main__':
    cv2.setNumThreads(0)

    opt = parse_args()
    print(opt)

    (Path(opt.out_path) / 'test').mkdir(parents=True, exist_ok=True)
    (Path(opt.out_path) / 'val').mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(opt.ckpt)

    opt.channels = checkpoint['opt'].channels if 'channels' in checkpoint['opt'].__dict__ else 965
    opt.n_channels = checkpoint['opt'].n_channels if 'n_channels' in checkpoint['opt'].__dict__ else abs(opt.channels)
    opt.n_classes = checkpoint['opt'].n_classes
    opt.arch = checkpoint['opt'].arch

    test_set = RealDataset(opt.real_path, opt.channels, split='test')
    test_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=False)

    val_set = RealDataset(opt.real_path, opt.channels, split='val')
    val_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=False)

    net = PowderNet(opt.arch, opt.n_channels, opt.n_classes)
    net = net.cuda()
    net.load_state_dict(checkpoint['state_dict'])

    log_file = open(Path(opt.out_path) / 'performance.txt', 'w')
    print(opt, file=log_file)
    msa, miou, miiou, msa_crf, miou_crf, miiou_crf = test(opt, test_loader, net, 'test')
    print('Test  Pre-CRF:  MSA: {}   mIoU: {}   miIoU: {}'.format(round(msa * 100, 1), round(miou * 100, 1), round(miiou * 100, 1)), file=log_file)
    print('Test  Post-CRF: MSA: {}   mIoU: {}   miIoU: {}'.format(round(msa_crf * 100, 1), round(miou_crf * 100, 1), round(miiou_crf * 100, 1)), file=log_file)
    msa, miou, miiou, msa_crf, miou_crf, miiou_crf = test(opt, val_loader, net, 'val')
    print('Val  Pre-CRF:  MSA: {}   mIoU: {}   miIoU: {}'.format(round(msa * 100, 1), round(miou * 100, 1), round(miiou * 100, 1)), file=log_file)
    print('Val  Post-CRF: MSA: {}   mIoU: {}   miIoU: {}'.format(round(msa_crf * 100, 1), round(miou_crf * 100, 1), round(miiou_crf * 100, 1)), file=log_file)
    print('Complete', file=log_file)
    log_file.close()
