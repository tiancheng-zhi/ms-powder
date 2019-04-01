import h5py
import numpy as np
import cv2

from pathlib import Path

if __name__ == '__main__':
    data_path = Path('../data/trainext/')
    real_path = Path('../real/')
    n_scenes = 64
    height = 160
    width = 280
    n_channels = 38
    h5f = h5py.File(str(Path(real_path / ('trainext.hdf5'))), 'w')
    dset_im = h5f.create_dataset('im', (n_scenes, height, width, n_channels), dtype='float32')
    dset_label = h5f.create_dataset('label', (n_scenes, height, width), dtype='uint8')
    lights = ['EiKOIncandescent250W', 'IIIWoodsHalogen500W', 'LowelProHalogen250W', 'WestinghouseIncandescent150W']

    for i in range(n_scenes):
        idx = str(i % (n_scenes // len(lights))).zfill(2)
        light = lights[i // (n_scenes // len(lights))]
        im_npz = np.load(data_path / light / 'scene' / (idx + '_scene.npz'))
        im = np.concatenate((im_npz['rgbn'].astype(np.float32), im_npz['swir'].astype(np.float32)), axis=2)
        label = cv2.imread(str(data_path / light / 'label' / (idx + '_label.png')), cv2.IMREAD_GRAYSCALE)
        dset_im[i, :, :, :] = im
        dset_label[i, :, :] = label
    h5f.close()
