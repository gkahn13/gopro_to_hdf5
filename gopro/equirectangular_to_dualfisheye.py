import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from utils import np_utils


class EquirectangularToDualfisheye(object):

    EQUI_SHAPE = (1920, 3840, 3)
    DUALFISHEYE_SHAPE = (1000, 2000, 3)
    FISHEYE_SHAPE = (450, 800, 3)

    def __init__(self):
        targ_u, targ_v = np.meshgrid(np.linspace(0., 1., self.DUALFISHEYE_SHAPE[1]),
                                     np.linspace(0., 1., self.DUALFISHEYE_SHAPE[0]))

        up, v = targ_u, targ_v

        u = 2.0 * (up - 0.5)
        mask = up < 0.5
        u = np.logical_not(mask) * u + mask * 2.0 * up

        self._keep_mask = ((u - 0.5) * (u - 0.5) + (v - 0.5) * (v - 0.5)) <= 0.25

        phi = np.arcsin(2.0 * (v - 0.5))
        u = 1.0 - u
        theta = np.arccos(np.clip(2.0 * (u - 0.5) / np.cos(phi), -1., 1.))

        theta = np.logical_not(mask) * theta + mask * (theta - np.pi)

        targ_theta, targ_phi = theta, phi

        src_u = 0.5 + 0.5 * (targ_theta / np.pi)
        src_v = 0.5 + (targ_phi / np.pi)

        self._remap = np.stack([self.EQUI_SHAPE[0] * src_v, self.EQUI_SHAPE[1] * src_u], axis=-1).astype(np.float32)

    def to_dualfisheye(self, im):
        assert tuple(im.shape) == self.EQUI_SHAPE

        im = np.roll(im, im.shape[1] // 4, axis=1)

        im_dualfish = cv2.remap(im,
                                self._remap[..., 1],
                                self._remap[..., 0],
                                interpolation=cv2.INTER_CUBIC)
        im_dualfish[np.logical_not(self._keep_mask)] = [0, 0, 0]
        return im_dualfish

    def to_rectified(self, im, plot=False):
        im_dualfish = self.to_dualfisheye(im)

        im_fish = im_dualfish[:, im_dualfish.shape[1]//2:]
        assert tuple(im_fish.shape[:2]) == (1000, 1000)

        # fx, fy, cx, cy = [499.2896289085027, 500.25930528321345, 500.39316178263533, 497.7380469112018]
        fx, fy, cx, cy = [500., 500., 500., 500.]
        K = np.array([[fx, 0., cx],
                      [0., fy, cy],
                      [0., 0., 1.]])
        D = 0.9 * np.array([[-0.14978641102360382, -0.012433353097853287, 0.01134655115827489, -0.0028248970180605]]).T
        balance = 1.0
        im_fish_rect = np_utils.imrectify_fisheye(im_fish, K, D, balance)

        im_fish_clip = im_fish_rect[275:-275, 100:-100]

        if plot:
            f, axes = plt.subplots(1, 3)
            axes[0].imshow(im_fish)
            axes[1].imshow(im_fish_rect)
            axes[2].imshow(im_fish_clip)
            plt.show()

        return im_fish_clip


if __name__ == '__main__':
    im_equi = np.array(Image.open('/home/gkahn/source/vrProjector/gopro/equirectangular/0001.jpg'))
    equi_to_fisheye = EquirectangularToDualfisheye()
    equi_to_fisheye.to_rectified(im_equi, plot=True)
