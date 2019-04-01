import argparse
import numpy as np
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Calibration')
    parser.add_argument('--data-path', type=str, default='../data/train')
    parser.add_argument('--out-path', type=str, default='../params')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_args()
    print(opt)

    Path(opt.out_path).mkdir(parents=True, exist_ok=True)

    lights = ['EiKOIncandescent250W', 'IIIWoodsHalogen500W', 'LowelProHalogen250W', 'WestinghouseIncandescent150W']
    n = 100
    h = 14
    w = 14
    c = 965
    valid_threshold = 20

    kappa_params = []
    for i in range(n):
        print(i)
        key = str(i).zfill(2)
        kappa_lights = []
        for lid, light in enumerate(lights):
            thick_path = Path(opt.data_path) / light / 'thick'
            thin_path = Path(opt.data_path) / light / 'thin'
            bg_path = Path(opt.data_path) / light / 'bg'
            thick = np.load(thick_path / (key + '_thick.npz'))
            thin = np.load(thin_path / (key + '_thin.npz'))
            bg = np.load(bg_path / (key + '_bg.npz'))
            thick = np.concatenate((thick['rgbn'], thick['swir']), 2)
            thin = np.concatenate((thin['rgbn'], thin['swir']), 2)
            bg = np.concatenate((bg['rgbn'], bg['swir']), 2)


            thick = np.mean(thick, (0, 1), keepdims=True)
            bg = np.mean(bg, (0, 1), keepdims=True)
            alpha = (thin - thick) / (bg - thick)

            # valid alpha selection
            alpha = alpha.reshape([h * w, c])
            alpha = np.clip(alpha, 0.01, 0.99)
            kt = -np.log(alpha)
            kappa = np.median(kt, axis=0)
            ratio = (kappa[:4].mean() + kappa[4:].mean()) / 2
            kappa = kappa / ratio
            print(kappa[:4], kappa[[5, -1]], kappa.max(), kappa.min())
            kappa_lights.append(kappa)
        kappa_params.append(kappa_lights)
    np.savez(Path(opt.out_path) / 'kappa_params.npz', params=kappa_params)
