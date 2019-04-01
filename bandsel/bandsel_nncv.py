import argparse
import numpy as np
import torch
import torch.nn.functional as F
import numba
import cv2
import time

from pathlib import Path


def cosine(a, b):
    y = b.unsqueeze(0)
    n_pixels = a.size()[0]
    batch_size = 1024
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


def sim_func(dist, query, database):
    if (dist == 'full') or (query.size()[1] <= 5):
        return cosine(query, database)
    elif dist == 'split':
        return cosine(query[:,:4], database[:,:4]) + cosine(query[:,4:], database[:,4:])
    else:
        assert(False)


def feat_eng(dist, raw):
    if dist == 'full' or dist == 'split':
        return raw
    elif dist == 'decouple':
        swir = raw[:, 4:]
        mean_swir = swir.mean(dim=1, keepdim=True)
        feat = torch.cat((mean_swir, raw), dim=1)
        return feat
    else:
        assert(False)


def parse_args():
    parser = argparse.ArgumentParser(description='NNCV Band Selection')
    parser.add_argument('--data-path', type=str, default='../data')
    parser.add_argument('--log-path', type=str, default='./bands')
    parser.add_argument('--n-sels', type=int, default=49)
    parser.add_argument('--dist', type=str, default='split', choices=['full', 'split'])
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_args()
    Path(opt.log_path).mkdir(parents=True, exist_ok=True)
    if opt.dist == 'split':
        log = open(Path(opt.log_path) / ('nncv.txt'), 'w')
    else:
        log = open(Path(opt.log_path) / ('nncv_{}.txt'.format(opt.dist)), 'w')
    print(opt)
    opt.lights = ['EiKOIncandescent250W', 'IIIWoodsHalogen500W', 'LowelProHalogen250W', 'WestinghouseIncandescent150W']
    opt.n_lights = len(opt.lights)
    opt.n_powders = 100
    opt.n_bgmats = 100
    opt.n_channels = 965
    opt.n_full_swir_channels = 961

    train_path = Path(opt.data_path) / 'train'

    y = []

    thick_list = np.zeros((opt.n_powders, opt.n_lights, opt.n_channels))
    for i in range(opt.n_powders):
        idx = str(i).zfill(2)
        for lid, light in enumerate(opt.lights):
            thick = np.load(train_path / light / 'thick' / (idx + '_thick.npz'))
            thick = np.concatenate((thick['rgbn'], thick['swir']), axis=2)
            thick = thick.mean((0, 1))
            thick_list[i, lid] = thick
            y.append(i)
    thick_list = thick_list.reshape((opt.n_powders * opt.n_lights, opt.n_channels))

    bgmat_list = np.zeros((opt.n_bgmats, opt.n_lights, opt.n_channels))
    for i in range(opt.n_bgmats):
        idx = str(i).zfill(2)
        for lid, light in enumerate(opt.lights):
            bgmat = np.load(train_path / light / 'bgmat' / (idx + '_bgmat.npz'))
            bgmat = np.concatenate((bgmat['rgbn'], bgmat['swir']), axis=2)
            bgmat = bgmat.mean((0, 1))
            bgmat_list[i, lid] = bgmat
            y.append(opt.n_powders)
    bgmat_list = bgmat_list.reshape((opt.n_bgmats * opt.n_lights, opt.n_channels))

    raw = np.concatenate((thick_list, bgmat_list), axis=0)
    raw = torch.from_numpy(raw).cuda()
    y = np.array(y)
    y = torch.from_numpy(y).cuda()

    selection = np.zeros(opt.n_channels, dtype=np.bool_)
    selection[0] = True
    selection[1] = True
    selection[2] = True
    selection[3] = True

    start_time = time.time()
    bands = []
    for i in range(opt.n_sels):
        best_acc = 0
        for j in range(opt.n_channels):
            if selection[j]:
                continue
            selection[j] = True
            selection_th = torch.from_numpy(selection.astype(np.uint8)).unsqueeze(0).cuda()
            x = torch.masked_select(raw, selection_th).view(raw.size()[0], -1)
            if i > 0:
                x = feat_eng(opt.dist, x)

            sims = sim_func(opt.dist, x, x)
            _, indices = torch.sort(sims, dim=1)
            acc = (y == y[indices[:,-2]]).cpu().numpy().astype(np.float32)

            acc = acc.reshape((opt.n_powders + opt.n_bgmats, opt.n_lights)).mean(axis=1)

            acc = (acc[:opt.n_powders].sum() + acc[opt.n_powders:].mean()) / (opt.n_powders + 1)
            if acc > best_acc:
                best_acc = acc
                best_sel = j
            selection[j] = False
        selection[best_sel] = True
        print(i, best_acc, best_sel - 4, round(time.time()-start_time))
        bands.append(best_sel - 4)

    print(bands)
    st = ''
    for i in bands:
        st = st + ',' + str(i)
    st = st[1:]
    print(st, file=log)
    log.close()

