import h5py
import numpy as np
import cv2

from pathlib import Path

if __name__ == '__main__':
    data_path_bg = Path('../data/train/')
    data_path_ext = Path('../data/trainext/')
    real_path = Path('../real/bgext/')
    real_path.mkdir(exist_ok=True, parents=True)
    n_scenes = 16
    height = 160
    width = 280
    n_channels = 38
    sel = [0, 1, 2, 3, 4, 5, 19, 34, 51, 77, 95, 127, 152, 342, 399, 401, 422, 434, 442, 469, 484, 487, 499, 538, 555, 588, 637, 664, 676, 683, 686, 750, 837, 879, 905, 934, 949, 964]
    lights = ['EiKOIncandescent250W', 'IIIWoodsHalogen500W', 'LowelProHalogen250W', 'WestinghouseIncandescent150W']
    for light in lights:
        h5f = h5py.File(str(Path(real_path / (light + '.hdf5'))), 'w')
        dset_im = h5f.create_dataset('im', (n_scenes * 2, height, width, n_channels), dtype='float32')
        for i in range(n_scenes):
            idx = str(i).zfill(2)
            im_npz = np.load(data_path_bg / light / 'bgscene' / (idx + '_bgscene.npz'))
            im = np.concatenate((im_npz['rgbn'].astype(np.float32), im_npz['swir'].astype(np.float32)), axis=2)
            dset_im[i, :, :, :] = im[:, :, sel]
        for i in range(n_scenes):
            idx = str(i).zfill(2)
            im_npz = np.load(data_path_ext / light / 'bgscene' / (idx + '_bgscene.npz'))
            im = np.concatenate((im_npz['rgbn'].astype(np.float32), im_npz['swir'].astype(np.float32)), axis=2)
            dset_im[n_scenes + i, :, :, :] = im
        h5f.close()
