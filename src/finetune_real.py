import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from cosine_scheduler import CosineLRWithRestarts
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import RealDataset 
from visualizer import Visualizer
from model import PowderNet
from utils import to_image, colormap, errormap
from adamw import AdamW
from model import get_1x_lr_params, get_10x_lr_params
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='PowderDetector')
    parser.add_argument('--real-path', type=str, default='../real')
    parser.add_argument('--params-path', type=str, default='../params')
    parser.add_argument('--out-path', type=str, default='./checkpoint')
    parser.add_argument('--channels', type=int, choices=[965, 4, -961, 0], default=0, help='x>0 select [:x]; x<0 select [x:]; x=0 see --bands')
    parser.add_argument('--bands', type=str, default=None)
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--blend', type=str, choices=['none', 'alpha', 'kappa'], default='kappa')
    parser.add_argument('--arch', type=str, choices=['deeplab'], default='deeplab')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n-epochs', type=int, default=24)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--decay', type=float, default=1e-4)
    parser.add_argument('--period', type=int, default=8)
    parser.add_argument('--t-mult', type=float, default=2)
    parser.add_argument('--vis-iter', type=int, default=0)
    parser.add_argument('--server', type=str, default='http://localhost')
    parser.add_argument('--env', type=str, default='main')
    opt = parser.parse_args()
    if opt.bands is not None:
        assert(opt.channels == 0)
        opt.channels = [int(i) for i in opt.bands.split(',')]
        opt.n_channels = len(opt.channels)
    else:
        opt.n_channels = abs(opt.channels)
    return opt


def train(opt, vis, epoch, train_loader, net, optimizer, scheduler):
    net = net.train()
    train_len = len(train_loader)
    start_time = time.time()
    scheduler.step()
    for iteration, batch in enumerate(train_loader):
        # Load Data
        im, label = batch
        im = im.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        # Forward Pass
        out = net(im)
        loss = F.cross_entropy(out, label)

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.batch_step()

        # Logging
        cur_time = time.time()
        loss_scalar = float(loss.cpu().detach().numpy())
        if iteration < opt.threads:
            print('{} [{}]({}/{}) AvgTime:{:>4} Loss:{:>4}'.format(opt.env, epoch, iteration, train_len, \
                                                                   round((cur_time - start_time) / (iteration + 1), 2), \
                                                                   round(loss_scalar, 4)))
            if iteration == opt.threads - 1:
                start_time = cur_time
        else:
            print('{} [{}]({}/{}) AvgTime:{:>4} Loss:{:>4}'.format(opt.env, epoch, iteration, train_len, \
                                                                   round((cur_time - start_time) / (iteration + 1 - opt.threads), 2), \
                                                                   round(loss_scalar, 4)))

        # Visualization
        vis.iteration.append(epoch + iteration / train_len)
        vis.nlogloss.append(-np.log(np.maximum(1e-6, loss_scalar)))
        vis.plot_loss()
        if opt.vis_iter <= 0 or iteration % opt.vis_iter > 0:
            continue
        prob, pred = torch.max(out, dim=1)
        vis_rgb = to_image(im[0, 0:3, :, :] * 0.5)
        vis_nir = to_image(im[0, 3:4, :, :] * 0.5)
        vis_swir1 = to_image(im[0, 4:5, :, :] * 0.5)
        vis_swir2 = to_image(im[0, -2:-1, :, :] * 0.5)
        vis_label = colormap(label[0].cpu().numpy())
        vis_pred = colormap(pred[0].cpu().numpy())
        vis_im = np.concatenate((np.concatenate((vis_label, vis_pred), axis=1), \
                                 np.concatenate((vis_rgb, vis_nir), axis=1), \
                                 np.concatenate((vis_swir1, vis_swir2), axis=1)), axis=2)
        vis.plot_image(vis_im, 0)


def test(opt, epoch, test_loader, net):
    if opt.channels == 965:
        bilateral_ch = [0,1,2,3,4,159,324,469,644,779,964]
    elif opt.channels == 4:
        bilateral_ch = [0,1,2,3]
    elif opt.channels == -961:
        bilateral_ch = [0,155,320,465,640,775,960]
    else:
        bilateral_ch = range(opt.n_channels)
    net = net.eval()
    test_len = len(test_loader)
    tp = np.zeros(opt.n_classes)
    fp = np.zeros(opt.n_classes)
    tp_crf = np.zeros(opt.n_classes)
    fp_crf = np.zeros(opt.n_classes)
    num = np.zeros(opt.n_classes)
    start_time = time.time()
    for iteration, batch in enumerate(test_loader):
        # Load Data
        im, label = batch
        im = im.cuda()
        label = label.cuda()

        # Forward Pass
        out = net(im)

        # Visualization
        prob = F.softmax(out, dim=1)
        _, pred = torch.max(prob, dim=1)

        bsize = pred.size()[0]

        for i in range(bsize):
            label_np = label[i].cpu().numpy()
            pred_np = pred[i].cpu().numpy()
            for c in range(opt.n_classes):
                mask = (label_np == c)
                tp[c] += ((pred_np == c) * mask).sum()
                fp[c] += ((pred_np == c) * (1 - mask)).sum()
                num[c] += mask.sum()

    iou = tp / (num + fp)
    miou = iou.mean()
    return miou


if __name__ == '__main__':
    cv2.setNumThreads(0)

    opt = parse_args()
    print(opt)

    Path(opt.out_path).mkdir(parents=True, exist_ok=True)

    train_set = RealDataset(opt.real_path, opt.channels, split='trainext', flip=True)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True, pin_memory=True)

    val_set = RealDataset(opt.real_path, opt.channels, split='val')
    val_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=1, shuffle=False)

    test_set = RealDataset(opt.real_path, opt.channels, split='test')
    test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)

    opt.n_classes = train_set.n_classes
    net = PowderNet(opt.arch, opt.n_channels, train_set.n_classes)
    net = net.cuda()
    optimizer = AdamW([{'params': get_1x_lr_params(net)}, {'params': get_10x_lr_params(net), 'lr': opt.lr * 10}], lr=opt.lr, weight_decay=opt.decay)
    scheduler = CosineLRWithRestarts(optimizer, opt.batch_size, len(train_set), opt.period, opt.t_mult)
    vis = Visualizer(server=opt.server, env=opt.env)
    start_epoch = 0
    if opt.resume is not None:
        checkpoint = torch.load(opt.resume)
        old_opt = checkpoint['opt']
        assert(old_opt.channels == opt.channels)
        assert(old_opt.bands == opt.bands)
        assert(old_opt.arch == opt.arch)
        assert(old_opt.blend == opt.blend)
        assert(old_opt.lr == opt.lr)
        assert(old_opt.decay == opt.decay)
        assert(old_opt.period == opt.period)
        assert(old_opt.t_mult == opt.t_mult)
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        vis.load_state_dict(checkpoint['vis'])
        start_epoch = checkpoint['epoch'] + 1
    elif opt.pretrain is not None:
        checkpoint = torch.load(opt.pretrain)
        old_opt = checkpoint['opt']
        assert(old_opt.channels == opt.channels)
        assert(old_opt.bands == opt.bands)
        assert(old_opt.arch == opt.arch)
        assert(old_opt.blend == opt.blend)
        net.load_state_dict(checkpoint['state_dict'])
    else:
        assert(False)

    for epoch in range(start_epoch, opt.n_epochs):
        train(opt, vis, epoch, train_loader, net, optimizer, scheduler)
        miou_val = test(opt, epoch, val_loader, net)
        miou_test = test(opt, epoch, test_loader, net)
        vis.epoch.append(epoch)
        vis.acc.append([miou_val, miou_test])
        vis.plot_acc()
        if (epoch + 1) % opt.period == 0:
            torch.save({'epoch': epoch, 'opt': opt, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),  'scheduler': scheduler.state_dict(), 'vis': vis.state_dict()}, Path(opt.out_path) / (str(epoch) + '.pth'))
        print('Val mIoU:', miou_val, ' Test mIoU:', miou_test)
