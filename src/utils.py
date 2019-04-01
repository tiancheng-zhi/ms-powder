import numpy as np
import torch


def cpu_np(tensor):
    return tensor.cpu().numpy()


def to_image(matrix):
    image = cpu_np(torch.clamp(matrix, 0, 1) * 255).astype(np.uint8)
    if matrix.size()[0] == 1:
        image = np.concatenate((image, image, image), 0)
        return image
    return image


def errormap(green, yellow, red, blue):
    err = np.zeros((3, green.shape[0], green.shape[1]), dtype=np.uint8)
    err[2, :, :][blue] = 255
    err[0, :, :][red] = 255
    err[0, :, :][yellow] = 255
    err[1, :, :][yellow] = 255
    err[2, :, :][yellow] = 0
    err[0, :, :][green] = 0
    err[1, :, :][green] = 255
    return err


def colormap(label):
    cm = []
    for r in [35, 90, 145, 200, 255]:
        for g in [35, 90, 145, 200, 255]:
            for b in [60, 125, 190, 255]:
                cm.append((r, g, b))
    cm.append((0, 0, 0))
    label_cm = np.stack((label, label, label), 0).astype(np.uint8)
    for c, color in enumerate(cm):
        mask = (label == c)
        label_cm[0, :, :][mask] = color[0]
        label_cm[1, :, :][mask] = color[1]
        label_cm[2, :, :][mask] = color[2]
    return label_cm
