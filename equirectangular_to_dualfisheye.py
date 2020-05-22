import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import np_utils

def equirectangular_to_dualfisheye(im_equi, shape_dualfish, remap=None, keep_mask=None,
                                   interpolation=cv2.INTER_CUBIC):
    assert im_equi.shape[1] / im_equi.shape[0] == 2
    assert shape_dualfish[1] / shape_dualfish[0] == 2

    im_equi = np.roll(im_equi, im_equi.shape[1] // 4, axis=1)

    if remap is None or keep_mask is None:
        targ_u, targ_v = np.meshgrid(np.linspace(0., 1., shape_dualfish[1]),
                                     np.linspace(0., 1., shape_dualfish[0]))

        up, v = targ_u, targ_v

        u = 2.0 * (up - 0.5)
        mask = up < 0.5
        u = np.logical_not(mask) * u + mask * 2.0 * up

        keep_mask = ((u - 0.5) * (u - 0.5) + (v - 0.5) * (v - 0.5)) <= 0.25

        phi = np.arcsin(2.0 * (v - 0.5))
        u = 1.0 - u
        theta = np.arccos(np.clip(2.0 * (u - 0.5) / np.cos(phi), -1., 1.))

        theta = np.logical_not(mask) * theta + mask * (theta - np.pi)

        targ_theta, targ_phi = theta, phi

        src_u = 0.5 + 0.5 * (targ_theta / np.pi)
        src_v = 0.5 + (targ_phi / np.pi)

        remap = np.stack([im_equi.shape[0] * src_v, im_equi.shape[1] * src_u], axis=-1).astype(np.float32)

    # just this part is 25Hz for 1024x2048
    im_dualfish = cv2.remap(im_equi,
                            remap[..., 1],
                            remap[..., 0],
                            interpolation=interpolation)
    im_dualfish[np.logical_not(keep_mask)] = [0, 0, 0]

    return im_dualfish, remap, keep_mask

def SLOW_equirectangular_to_dualfisheye(im_equi, shape_dualfish, remap=None, keep_mask=None,
                                   interpolation=cv2.INTER_CUBIC):
    assert im_equi.shape[1] / im_equi.shape[0] == 2
    assert shape_dualfish[1] / shape_dualfish[0] == 2

    im_equi = np.roll(im_equi, im_equi.shape[1] // 4, axis=1)

    def angular_position(up, v):
        # correct for hemisphere
        if up >= 0.5:
            u = 2.0 * (up - 0.5)
        else:
            u = 2.0 * up

        # ignore points outside of circles
        if ((u - 0.5) * (u - 0.5) + (v - 0.5) * (v - 0.5)) > 0.25:
            return None, None

        # v: 0..1-> vp: -1..1
        phi = np.arcsin(2.0 * (v - 0.5))

        # u = math.cos(phi)*math.cos(theta)
        # u: 0..1 -> upp: -1..1
        u = 1.0 - u
        theta = np.arccos(np.clip(2.0 * (u - 0.5) / np.cos(phi), -1., 1.))

        if up < 0.5:
            theta = theta - np.pi

        return (theta, phi)

    if remap is None or keep_mask is None:
        remap = np.zeros(list(shape_dualfish) + [2], dtype=np.float32)
        keep_mask = np.zeros(shape_dualfish, dtype=np.bool)
        for x in range(shape_dualfish[1]):
            for y in range(shape_dualfish[0]):
                targ_u = float(x) / float(shape_dualfish[1])
                targ_v = float(y) / float(shape_dualfish[0])
                targ_theta, targ_phi = angular_position(targ_u, targ_v)

                if targ_theta is None or targ_phi is None:
                    continue
                keep_mask[y, x] = True

                # theta: -pi..pi -> u: 0..1
                src_u = 0.5 + 0.5 * (targ_theta / np.pi)
                # phi: -pi/2..pi/2 -> v: 0..1
                src_v = 0.5 + (targ_phi / np.pi)

                remap[y, x] = [im_equi.shape[0] * src_v,
                             im_equi.shape[1] * src_u]

    # just this part is 25Hz for 1024x2048
    im_dualfish = cv2.remap(im_equi,
                            remap[..., 1],
                            remap[..., 0],
                            interpolation=interpolation)
    im_dualfish[np.logical_not(keep_mask)] = [0, 0, 0]

    return im_dualfish, remap, keep_mask


def equirectangular_to_fisheye(im_equi, size, **kwargs):
    width = int(size)
    height = int(size * 9. / 16.)
    assert abs(height / float(width) - 9./16.) < 1e-4

    shape_dualfish = [size, 2 * size]
    im_dualfish, remap, keep_mask = equirectangular_to_dualfisheye(im_equi, shape_dualfish, **kwargs)
    im_fish = im_dualfish[:, size:]


    # clip = (size - height) / 2.
    # clip_top = int(np.ceil(clip))
    # clip_bottom = int(np.floor(clip))
    # im_fish_clip = im_fish[clip_top:-clip_bottom]
    #
    # assert tuple(im_fish_clip.shape) == (height, width, 3)
    #
    #
    # fx, fy, cx, cy = 0.5*width, 0.5*height, 0.5*width, 0.5*height
    # K = np.array([[fx, 0., cx],
    #               [0., fy, cy],
    #               [0., 0., 1.]])
    # D = 0 * np.array([[-0.038483, -0.010456, 0.003930, -0.001007]]).T
    # balance = 0.5
    # im_fish_clip = np_utils.imrectify_fisheye(im_fish_clip, K, D, balance)


    fx, fy, cx, cy = 0.5*size, 0.5*size, 0.5*size, 0.5*size
    # fx, fy, cx, cy = 160., 160., 160., 160.
    K = np.array([[fx, 0., cx],
                  [0., fy, cy],
                  [0., 0., 1.]])
    D =  2 * np.array([[-0.038483, -0.010456, 0.003930, -0.001007]]).T
    balance = 0.5
    im_fish_clip = np_utils.imrectify_fisheye(im_fish, K, D, balance)


    print('plotting!')
    f, axes = plt.subplots(1, 2)
    axes[0].imshow(im_fish)
    axes[1].imshow(im_fish_clip)
    plt.show()
    import sys; sys.exit(0)



    return im_fish_clip, remap, keep_mask




im_equi = np.array(Image.open('/home/gkahn/source/vrProjector/gopro/equirectangular/0001.jpg'))
im_equi = np_utils.imresize(im_equi, [480, 960, 3])
im_dualfish, remap, keep_mask = equirectangular_to_fisheye(im_equi, size=960)

# import time
# start = time.time()
# for _ in range(100):
#     im_dualfish, remap, keep_mask = equirectangular_to_dualfisheye(im_equi, [1024, 2048], remap, keep_mask,
#                                                                    cv2.INTER_CUBIC)
# elapsed = time.time() - start
# print(elapsed / 100.)
# print(100. / elapsed)

print('Showing')
import matplotlib.pyplot as plt
f, (ax0, ax1) = plt.subplots(2, 1)
ax0.imshow(im_equi)
ax1.imshow(im_dualfish)
plt.show()
