import h5py
import numpy as np
import cv2

from pathlib import Path

if __name__ == '__main__':
    data_path = Path('../data/train/')
    real_path = Path('../real/bg/')
    n_scenes = 16
    height = 160
    width = 280
    n_channels = 965
    lights = ['EiKOIncandescent250W', 'IIIWoodsHalogen500W', 'LowelProHalogen250W', 'WestinghouseIncandescent150W']

    for light in lights:
        h5f = h5py.File(str(Path(real_path / (light + '.hdf5'))), 'w')
        dset_im = h5f.create_dataset('im', (n_scenes, height, width, n_channels), dtype='float32')
        for i in range(n_scenes):
            idx = str(i).zfill(2)
            im_npz = np.load(data_path / light / 'bgscene' / (idx + '_bgscene.npz'))
            im = np.concatenate((im_npz['rgbn'].astype(np.float32), im_npz['swir'].astype(np.float32)), axis=2)
            dset_im[i, :, :, :] = im
        h5f.close()
