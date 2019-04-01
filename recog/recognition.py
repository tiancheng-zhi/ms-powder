import argparse
import collections
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import time

from pathlib import Path


def cosine(a, b):
    y = b.unsqueeze(0)
    n_pixels = a.size()[0]
    batch_size = 1000
    if n_pixels % batch_size == 0:
        n_batches = n_pixels // batch_size
    else:
        n_batches = n_pixels // batch_size + 1
    sim = torch.zeros((n_pixels, b.size()[0]), dtype=torch.double).cuda()
    for batch_idx in range(n_batches):
        bs = batch_idx * batch_size
        be = min(n_pixels, (batch_idx + 1) * batch_size)
        x = a[bs:be, :].unsqueeze(1)
        sim[bs:be, :] = F.cosine_similarity(x, y, dim=2)
    return sim


def sim_func(opt, query, database):
    if opt.dist == 'full':
        return cosine(query, database)
    elif opt.dist == 'split':
        return cosine(query[:, :opt.n_rgbns], database[:, :opt.n_rgbns]) + cosine(query[:, opt.n_rgbns:], database[:, opt.n_rgbns:])
    else:
        assert(False)


def sims2pred(n_lights, sims):
    votes = sims.argmax(dim=1) // n_lights
    votes = votes.cpu().numpy()
    counts = collections.Counter(votes)
    pred = [i[0] for i in counts.most_common()]
    return pred


def match_powder_none(opt, database, scene):
    n_pixels = scene.size()[0]
    query = scene.view((n_pixels, -1))
    sims = sim_func(opt, query, database)
    return sims


def match_powder_kappa(opt, database, scene, bg, kappa):
    n_database = database.size()[0]
    n_pixels = scene.size()[0]
    n_channels = scene.size()[1]

    eta = torch.linspace(0, opt.eta_max, opt.n_etas, dtype=torch.double).cuda()
    alpha = eta.unsqueeze(0).unsqueeze(2) ** kappa.unsqueeze(1)

    # n_pixels * n_database * n_etas * n_channels
    bg = bg.unsqueeze(1).unsqueeze(2)

    # n_database * n_etas * n_channels
    database = database.unsqueeze(1) * (1 - alpha)

    batch_size = 64000 // n_channels
    sims = torch.zeros((n_pixels, n_database), dtype=torch.double).cuda()
    if n_database % batch_size == 0:
        n_batches = n_database // batch_size
    else:
        n_batches = n_database // batch_size + 1

    for p in range(n_pixels):
        query = scene[p:p+1, :]
        db = database + bg[p, :, :, :] * alpha
        for batch_idx in range(n_batches):
            bs = batch_idx * batch_size
            be = min(n_database, (batch_idx + 1) * batch_size)
            cur_database = db[bs:be, :, :].reshape(((be - bs) * opt.n_etas, -1))
            cur_sims = sim_func(opt, query, cur_database)
            cur_sims, _ = cur_sims.view((be - bs, opt.n_etas)).max(1)
            sims[p, bs:be] = cur_sims

    return sims


def parse_args():
   parser = argparse.ArgumentParser(description='Recognition with known mask')
   parser.add_argument('--data-path', type=str, default='../data')
   parser.add_argument('--log-path', type=str, default='./log')
   parser.add_argument('--sel-path', type=str, default='../bandsel/bands')
   parser.add_argument('--bg', type=str, choices=['gt', 'inpaint'], default='inpaint')
   parser.add_argument('--blend', type=str, choices=['none', 'alpha', 'kappa'], default='kappa')
   parser.add_argument('--eta-max', type=float, default=0.9)
   parser.add_argument('--n-etas', type=int, default=10)
   parser.add_argument('--kappa-params', type=str, default='../params/kappa_params.npz')
   parser.add_argument('--n-swirs', type=int, default=4)
   parser.add_argument('--n-rgbns', type=int, default=4, choices=[0, 1, 3, 4])
   parser.add_argument('--sel', type=str, default='nncv', choices=['nncv', 'grid', 'mvpca', 'rs'])
   parser.add_argument('--dist', type=str, default='split', choices=['full', 'split'])
   parser.add_argument('--set', type=str, default='test', choices=['test', 'val'])
   opt = parser.parse_args()

   assert(opt.n_rgbns + opt.n_swirs > 1)
   if opt.n_rgbns <= 1 or opt.n_swirs <= 1:
       assert(opt.dist == 'full')

   return opt


if __name__ == '__main__':
    opt = parse_args()

    Path(opt.log_path).mkdir(parents=True, exist_ok=True)
    log_fname = Path(opt.log_path) / ('{}_{}_{}_{}_{}.txt'.format(opt.set, opt.bg, opt.blend, opt.n_swirs, opt.n_rgbns))
    assert(not log_fname.is_file())

    lights = ['EiKOIncandescent250W', 'IIIWoodsHalogen500W', 'LowelProHalogen250W', 'WestinghouseIncandescent150W']
    n_lights = len(lights)
    n_powders = 100
    n_scenes = 32

    if opt.n_rgbns == 4:
        rgbn_channels = [0, 1, 2, 3]
    elif opt.n_rgbns == 3:
        rgbn_channels = [0, 1, 2]
    elif opt.n_rgbns == 1:
        rgbn_channels = [3]
    else:
        rgbn_channels = []

    all_channels = rgbn_channels.copy()
    swir_channels = []
    if opt.n_swirs > 0:
        if opt.sel == 'grid':
            assert(int(np.sqrt(opt.n_swirs))**2 == opt.n_swirs)
            if opt.n_swirs == 1:
                swir_channels.append(480)
            else:
                decimation = int(30 // (np.sqrt(opt.n_swirs) - 1))
                for i in range(0, 31, decimation):
                    for j in range(0, 31, decimation):
                        swir_channels.append(i * 31 + j)
        else:
            sel_file = open(Path(opt.sel_path) / (opt.sel + '.txt'), 'r')
            splited = sel_file.readlines()[-1].strip().split(',')
            sel_file.close()
            for i in splited[:opt.n_swirs]:
                swir_channels.append(int(i))
        assert(len(swir_channels) == opt.n_swirs)
        for i in swir_channels:
            all_channels.append(i + 4)

    n_channels = opt.n_rgbns + opt.n_swirs

    log_file = open(log_fname, 'w')
    print(opt)
    print(opt, file=log_file)
    print(swir_channels)
    print(swir_channels, file=log_file)

    train_path = Path(opt.data_path) / 'train'
    test_path = Path(opt.data_path) / opt.set

    scene_path = test_path / 'scene'
    bgscene_path = test_path / 'bgscene'
    label_path = test_path / 'label'

    thick_list = np.zeros((n_powders, n_lights, n_channels))
    for i in range(n_powders):
        idx = str(i).zfill(2)
        for lid, light in enumerate(lights):
            thick = np.load(train_path / light / 'thick' / (idx + '_thick.npz'))
            thick = np.concatenate((thick['rgbn'][:, :, rgbn_channels], thick['swir'][:, :, swir_channels]), axis=2)
            thick = thick.mean((0, 1))
            thick_list[i, lid] = thick
    thick_list = thick_list.reshape((n_powders * n_lights, n_channels))
    thick_list = torch.from_numpy(thick_list).cuda()

    if opt.blend == 'alpha':
        kappa = torch.ones((n_powders * n_lights, n_channels)).double().cuda()
    elif opt.blend == 'kappa':
        kappa_params = np.load(opt.kappa_params)
        kappa = kappa_params['params'][:, :, all_channels].reshape((n_powders * n_lights, n_channels))
        kappa = torch.from_numpy(kappa).cuda()

    acc_top1 = []
    acc_top3 = []
    start_time = time.time()
    for i in range(n_scenes):
        idx = str(i).zfill(2)
        print()
        print('scene', idx)

        print(file=log_file)
        print('scene', idx, file=log_file)

        scene = np.load(scene_path / (idx + '_scene.npz'))
        scene = np.concatenate((scene['rgbn'][:, :, rgbn_channels], scene['swir'][:, :, swir_channels]), axis=2)
        label = cv2.imread(str(label_path / (idx + '_label.png')), cv2.IMREAD_GRAYSCALE)
        if opt.bg == 'inpaint':
            mask = (label < 255).astype(np.uint8) * 255
            bgscene = scene.copy()
            for c in range(n_channels):
                scene_max = scene[mask == 255, c].max()
                bgscene[:, :, c] = (cv2.inpaint((scene[:, :, c] / scene_max * 65535).astype(np.uint16), mask, 3, cv2.INPAINT_TELEA)).astype(scene.dtype) * scene_max / 65535
        elif opt.bg == 'gt':
            bgscene = np.load(bgscene_path / (idx + '_bgscene.npz'))
            bgscene = np.concatenate((bgscene['rgbn'][:, :, rgbn_channels], bgscene['swir'][:, :, swir_channels]), axis=2)
        else:
            assert(False)
        for powder in range(n_powders):
            mask = (label == powder)
            if mask.any():
                print('powder', powder)
                print('powder', powder, file=log_file)
                scene_list = scene[mask, :]
                bgscene_list = bgscene[mask, :]
                scene_list = torch.from_numpy(scene_list).cuda()
                bgscene_list = torch.from_numpy(bgscene_list).cuda()
                if opt.blend == 'none':
                    sims = match_powder_none(opt, thick_list, scene_list)
                else:
                    sims = match_powder_kappa(opt, thick_list, scene_list, bgscene_list, kappa)
                pred = sims2pred(n_lights, sims)
                top1 = (powder in pred[:1])
                top3 = (powder in pred[:3])
                acc_top1.append(top1)
                acc_top3.append(top3)
                print(pred)
                print(top1, top3)
                print('Acc:', np.mean(acc_top1), np.mean(acc_top3))
                print('No.', len(acc_top1), ' ', (time.time() - start_time) / len(acc_top1), 's')
                print(pred, file=log_file)
                print(top1, top3, file=log_file)
                print('Acc:', np.mean(acc_top1), np.mean(acc_top3), file=log_file)
                print('No.', len(acc_top1), ' ', (time.time() - start_time) / len(acc_top1), 's', file=log_file)
    print(np.mean(acc_top1), np.mean(acc_top3))
    print(np.mean(acc_top1), np.mean(acc_top3), file=log_file)
    print('Complete', file=log_file)
    log_file.close()
