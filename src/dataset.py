import numpy as np
import h5py
import torch.utils.data as data
import torch
import cv2
import random
import scipy.special
from pathlib import Path


class SyntheticDataset(data.Dataset):

    def __init__(self, data_path, params_path, blend, channels):
        super(SyntheticDataset, self).__init__()
        assert(blend in ['none', 'alpha', 'kappa'])
        self.data_path = Path(data_path)
        self.blend = blend
        self.lights = ['EiKOIncandescent250W', 'IIIWoodsHalogen500W', 'LowelProHalogen250W', 'WestinghouseIncandescent150W']
        self.n_lights = len(self.lights)
        self.n_powders = 100
        self.height = 160
        self.width = 280
        self.channels = channels

        if type(channels) is int:
            self.channel = abs(channels)
            if channels > 0:
                self.ch_begin = 0
                self.ch_end = channels
            else:
                self.ch_begin = 965 + channels
                self.ch_end = 965
        else:
            self.channel = len(channels)
            self.ch_begin = None
            self.ch_end = None

        self.thickness_threshold = 0.1
        self.n_classes = 100 + 1
        self.n_per_light = 1000
        self.thick_sigma = 0.1
        self.shad_sigma = 0.1
        self.brdf_sigma = 0.1
        if blend == 'kappa':
            if self.ch_begin is None:
                self.kappa = np.load(Path(params_path) / 'kappa_params.npz')['params'][:, :, self.channels]
            else:
                self.kappa = np.load(Path(params_path) / 'kappa_params.npz')['params'][:, :, self.ch_begin:self.ch_end]
        else:
            self.kappa = None

    def __getitem__(self, index):
        lid = index // self.n_per_light
        light = self.lights[lid]
        powder_idx = index % self.n_per_light
        bg_idx = random.randint(0, self.n_per_light - 1)
        h5file = h5py.File(self.data_path / (light + '.hdf5'), 'r')
        if self.ch_begin is None:
            bg = h5file['bg'][bg_idx, :, :, self.channels].astype(np.float32)
            powder = h5file['powder'][powder_idx, :, :, self.channels].astype(np.float32)
        else:
            bg = h5file['bg'][bg_idx, :, :, self.ch_begin:self.ch_end].astype(np.float32)
            powder = h5file['powder'][powder_idx, :, :, self.ch_begin:self.ch_end].astype(np.float32)
        shading = h5file['shading'][bg_idx].astype(np.float32)
        label = h5file['label'][powder_idx]
        thickness = h5file['thickness'][powder_idx].astype(np.float32)
        h5file.close()

        if random.randint(0, 1) == 1:
            bg = np.fliplr(bg)
            shading = np.fliplr(shading)
        if random.randint(0, 1) == 1:
            bg = np.flipud(bg)
            shading = np.flipud(shading)
        if random.randint(0, 1) == 1:
            powder = np.fliplr(powder)
            label = np.fliplr(label)
            thickness = np.fliplr(thickness)
        if random.randint(0, 1) == 1:
            powder = np.flipud(powder)
            label = np.flipud(label)
            thickness = np.flipud(thickness)

        for i in range(self.n_powders):
            mask = (label == i)
            thickness[mask] = thickness[mask] * self.exp_gauss(self.thick_sigma)
            powder[mask] = powder[mask] * self.exp_gauss(self.brdf_sigma)
        label[thickness < self.thickness_threshold] = 255

        if self.blend == 'none':
            thickness = (thickness >= self.thickness_threshold).astype(np.float32)
            thickness = thickness[:, :, np.newaxis]
            alpha = 1 - thickness
        elif self.blend == 'alpha':
            thickness[thickness > 1] = 1
            thickness = thickness[:, :, np.newaxis]
            alpha = 1 - thickness
        elif self.blend == 'kappa':
            thickness[thickness > 1] = 1
            thickness = thickness[:, :, np.newaxis]
            alpha = np.ones((self.height, self.width, self.channel), dtype=np.float32)
            for i in range(self.n_powders):
                mask = (label == i)
                alpha[mask, :] = (1 - thickness[mask, :]) ** self.kappa[i, lid, :][np.newaxis, :]
        im = alpha * bg +  (1 - alpha) * powder
        med_shad = np.median(shading)
        im = im * shading[:, :, np.newaxis] / med_shad
        im = im *  self.exp_gauss(self.shad_sigma)

        im = im.transpose([2, 0, 1])
        label[label == 255] = self.n_classes - 1
        label = label.astype(np.int64)
        return im, label

    def exp_gauss(self, sigma):
        return np.exp(random.gauss(0, sigma))

    def __len__(self):
        return self.n_lights * self.n_per_light


class RealDataset(data.Dataset):

    def __init__(self, data_path, channels, split, flip=False):
        super(RealDataset, self).__init__()
        self.data_path = Path(data_path)
        self.n_classes = 100 + 1
        self.split = split
        self.flip = flip

        if split == 'trainext' or split == 'testext':
            assert(type(channels) is not int)
            self.n_images = 64
            self.channels = self.chmap(channels)
            self.channel = len(self.channels)
            self.ch_begin = None
            self.ch_end = None
        else:
            self.n_images = 32
            self.channels = channels
            if type(channels) is int:
                self.channel = abs(channels)
                if channels > 0:
                    self.ch_begin = 0
                    self.ch_end = channels
                else:
                    self.ch_begin = 965 + channels
                    self.ch_end = 965
            else:
                self.channel = len(channels)
                self.ch_begin = None
                self.ch_end = None

    def __getitem__(self, index):
        h5file = h5py.File(self.data_path / (self.split + '.hdf5'), 'r')
        if self.ch_begin is None:
            im = h5file['im'][index, :, :, self.channels].astype(np.float32)
        else:
            im = h5file['im'][index, :, :, self.ch_begin:self.ch_end].astype(np.float32)
        label = h5file['label'][index]
        h5file.close()
        
        if self.flip:
            if random.randint(0, 1) == 1:
                im = np.fliplr(im)
                label = np.fliplr(label)
            if random.randint(0, 1) == 1:
                im = np.flipud(im)
                label = np.flipud(label)
        
        im = im.transpose([2, 0, 1]).copy()
        label[label == 255] = self.n_classes - 1
        label = label.astype(np.int64).copy()
        return im, label

    def __len__(self):
        return self.n_images

    def chmap(self, channels):
        ch_list = [0, 1, 2, 3, 4, 5, 19, 34, 51, 77, 95, 127, 152, 342, 399, 401, 422, 434, 442, 469, 484, 487, 499, 538, 555, 588, 637, 664, 676, 683, 686, 750, 837, 879, 905, 934, 949, 964]
        mapped = []
        for i, ch in enumerate(ch_list):
            if ch in list(channels):
                mapped.append(i)
        print(channels, mapped)
        assert(len(channels) == len(mapped))
        return mapped


class HalfHalfDataset(data.Dataset):

    def __init__(self, real_path, syn_path, params_path, blend, channels, split):
        super(HalfHalfDataset, self).__init__()
        assert(blend in ['none', 'alpha', 'kappa'])
        self.real_path = Path(real_path) / split
        self.syn_path = Path(syn_path)
        self.blend = blend
        self.lights = ['EiKOIncandescent250W', 'IIIWoodsHalogen500W', 'LowelProHalogen250W', 'WestinghouseIncandescent150W']
        self.n_lights = len(self.lights)
        self.n_powders = 100
        self.height = 160
        self.width = 280
        self.channels = channels
        if split == 'bgext':
            assert(type(channels) is not int)
            self.n_bg_per_light = 32
            self.bg_channels = self.chmap(channels)
            self.channel = len(self.channels)
            self.ch_begin = None
            self.ch_end = None
        else:
            self.bg_channels = self.channels
            self.n_bg_per_light = 16
            if type(channels) is int:
                self.channel = abs(channels)
                if channels > 0:
                    self.ch_begin = 0
                    self.ch_end = channels
                else:
                    self.ch_begin = 965 + channels
                    self.ch_end = 965
            else:
                self.channel = len(channels)
                self.ch_begin = None
                self.ch_end = None
        self.thickness_threshold = 0.1
        self.n_classes = 100 + 1
        self.n_powder_per_light = 1000
        self.thick_sigma = 0.1
        self.brdf_sigma = 0.15
        if blend == 'kappa':
            if self.ch_begin is None:
                self.kappa = np.load(Path(params_path) / 'kappa_params.npz')['params'][:, :, self.channels]
            else:
                self.kappa = np.load(Path(params_path) / 'kappa_params.npz')['params'][:, :, self.ch_begin:self.ch_end]
        else:
            self.kappa = None

    def __getitem__(self, index):
        lid = index // self.n_powder_per_light
        light = self.lights[lid]
        powder_idx = index % self.n_powder_per_light
        bg_idx = random.randint(0, self.n_bg_per_light - 1)
        h5file = h5py.File(self.real_path / (light + '.hdf5'), 'r')
        if self.ch_begin is None:
            bg = h5file['im'][bg_idx, :, :, self.bg_channels].astype(np.float32)
        else:
            bg = h5file['im'][bg_idx, :, :, self.ch_begin:self.ch_end].astype(np.float32)
        h5file.close()
        h5file = h5py.File(self.syn_path / (light + '.hdf5'), 'r')
        if self.ch_begin is None:
            powder = h5file['powder'][powder_idx, :, :, self.channels].astype(np.float32)
        else:
            powder = h5file['powder'][powder_idx, :, :, self.ch_begin:self.ch_end].astype(np.float32)
        label = h5file['label'][powder_idx]
        thickness = h5file['thickness'][powder_idx].astype(np.float32)
        h5file.close()

        if random.randint(0, 1) == 1:
            bg = np.fliplr(bg)
        if random.randint(0, 1) == 1:
            bg = np.flipud(bg)
        if random.randint(0, 1) == 1:
            powder = np.fliplr(powder)
            label = np.fliplr(label)
            thickness = np.fliplr(thickness)
        if random.randint(0, 1) == 1:
            powder = np.flipud(powder)
            label = np.flipud(label)
            thickness = np.flipud(thickness)

        for i in range(self.n_powders):
            mask = (label == i)
            thickness[mask] = thickness[mask] * self.exp_gauss(self.thick_sigma)
            powder[mask] = powder[mask] * self.exp_gauss(self.brdf_sigma)
        label[thickness < self.thickness_threshold] = 255

        if self.blend == 'none':
            thickness = (thickness >= self.thickness_threshold).astype(np.float32)
            thickness = thickness[:, :, np.newaxis]
            alpha = 1 - thickness
        elif self.blend == 'alpha':
            thickness[thickness > 1] = 1
            thickness = thickness[:, :, np.newaxis]
            alpha = 1 - thickness
        elif self.blend == 'kappa':
            thickness[thickness > 1] = 1
            thickness = thickness[:, :, np.newaxis]
            alpha = np.ones((self.height, self.width, self.channel), dtype=np.float32)
            for i in range(self.n_powders):
                mask = (label == i)
                alpha[mask, :] = (1 - thickness[mask, :]) ** self.kappa[i, lid, :][np.newaxis, :]
        im = alpha * bg +  (1 - alpha) * powder

        im = im.transpose([2, 0, 1])
        label[label == 255] = self.n_classes - 1
        label = label.astype(np.int64)
        return im, label

    def exp_gauss(self, sigma):
        return np.exp(random.gauss(0, sigma))

    def __len__(self):
        return self.n_lights * self.n_powder_per_light

    def chmap(self, channels):
        ch_list = [0, 1, 2, 3, 4, 5, 19, 34, 51, 77, 95, 127, 152, 342, 399, 401, 422, 434, 442, 469, 484, 487, 499, 538, 555, 588, 637, 664, 676, 683, 686, 750, 837, 879, 905, 934, 949, 964]
        mapped = []
        for i, ch in enumerate(ch_list):
            if ch in list(channels):
                mapped.append(i)
        print(channels, mapped)
        assert(len(channels) == len(mapped))
        return mapped
