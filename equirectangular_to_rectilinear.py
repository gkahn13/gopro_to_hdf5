# adapted from https://github.com/NitishMutha/equirectangular-toolbox

import numpy as np


class EquirectangularToRectilinear(object):
    PI = np.pi
    PI_2 = np.pi * 0.5
    PI2 = np.pi * 2.0

    def __init__(self, frame_shape, shape, center_point=[0.5, 0.5], fov=0.45):
        # shapes must be width = 2 * height
        assert frame_shape[1] // frame_shape[0] == 2
        assert frame_shape[1] % frame_shape[0] == 0
        assert shape[1] // shape[0] == 2
        assert shape[1] % shape[0] == 0

        self._init_ops()

        self._fov = [fov, fov]
        self._frame_height, self._frame_width, self._frame_channel = frame_shape
        self._height, self._width = shape[:2]
        self._center_point = np.array(center_point)
        self._screen_points = self._get_screen_img()

        self._cp = self._get_coord_rad(center_point=self._center_point, is_center_pt=True)
        converted_screen_coord = self._get_coord_rad(is_center_pt=False)
        screen_coord = self._calc_spherical_to_gnomonic(converted_screen_coord)

        uf = np.mod(screen_coord[:, 0], 1) * self._frame_width  # long - width
        vf = np.mod(screen_coord[:, 1], 1) * self._frame_height  # lat - height

        x0float = self._floor_op(uf)  # coord of pixel to bottom left
        y0float = self._floor_op(vf)
        x2float = x0float + 1  # coords of pixel to top right
        y2float = y0float + 1

        x0 = self._cast_op(x0float, np.int32)
        x2 = self._cast_op(x2float, np.int32)
        y0 = self._cast_op(y0float, np.int32)
        y2 = self._cast_op(y2float, np.int32)

        base_y0 = y0 * self._frame_width
        base_y2 = y2 * self._frame_width

        self._A_idx = base_y0 + x0
        self._B_idx = base_y2 + x0
        self._C_idx = base_y0 + x2
        self._D_idx = base_y2 + x2

        wa = (x2float - uf) * (y2float - vf)
        wb = (x2float - uf) * (vf - y0float)
        wc = (uf - x0float) * (y2float - vf)
        wd = (uf - x0float) * (vf - y0float)

        self._wa = self._cast_op(self._tile_op(wa[:, None], (1, self._frame_channel)), np.float32)
        self._wb = self._cast_op(self._tile_op(wb[:, None], (1, self._frame_channel)), np.float32)
        self._wc = self._cast_op(self._tile_op(wc[:, None], (1, self._frame_channel)), np.float32)
        self._wd = self._cast_op(self._tile_op(wd[:, None], (1, self._frame_channel)), np.float32)

    def _init_ops(self):
        self._mod_op = np.mod
        self._floor_op = np.floor
        self._cast_op = lambda x, dtype: x.astype(dtype)
        self._tile_op = np.tile
        self._shape_op = lambda x: x.shape
        self._reshape_op = np.reshape
        self._gather_op = np.take

    def _get_coord_rad(self, is_center_pt, center_point=None):
        if is_center_pt:
            coord_rad = (center_point * 2 - 1) * np.array([self.PI, self.PI_2])
        else:
            coord_rad = (self._screen_points * 2 - 1) * np.array([self.PI, self.PI_2]) * self._fov
        return coord_rad

    def _get_screen_img(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, self._width), np.linspace(0, 1, self._height))
        return np.array([xx.ravel(), yy.ravel()]).T

    def _calc_spherical_to_gnomonic(self, converted_screen_coord):
        x = converted_screen_coord[:, 0]
        y = converted_screen_coord[:, 1]

        rou = np.sqrt(x * x + y * y)
        c = np.arctan(rou)
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        lat = np.arcsin(cos_c * np.sin(self._cp[1]) + (y * sin_c * np.cos(self._cp[1])) / rou)
        lon = self._cp[0] + np.arctan2(x * sin_c,
                                       rou * np.cos(self._cp[1]) * cos_c - y * np.sin(self._cp[1]) * sin_c)

        lat = (lat / self.PI_2 + 1.) * 0.5
        lon = (lon / self.PI + 1.) * 0.5

        return np.array([lon, lat]).T

    def _bilinear_interpolation(self, frame):
        flat_img = self._reshape_op(frame, [-1, self._frame_height * self._frame_width, self._frame_channel])

        A = self._gather_op(flat_img, self._A_idx, axis=1)
        B = self._gather_op(flat_img, self._B_idx, axis=1)
        C = self._gather_op(flat_img, self._C_idx, axis=1)
        D = self._gather_op(flat_img, self._D_idx, axis=1)

        # interpolate
        AA = self._cast_op(A, np.float32) * self._wa
        BB = self._cast_op(B, np.float32) * self._wb
        CC = self._cast_op(C, np.float32) * self._wc
        DD = self._cast_op(D, np.float32) * self._wd

        nfov = self._reshape_op(self._cast_op(AA + BB + CC + DD, np.uint8),
                                [-1, self._height, self._width, self._frame_channel])
        return nfov

    def to_rectilinear(self, frame):

        if len(self._shape_op(frame)) == 3:
            assert self._frame_channel == 1
            frame = frame[..., None]

        assert tuple(self._shape_op(frame)[1:]) == (self._frame_height, self._frame_width, self._frame_channel)

        frame_rect = self._bilinear_interpolation(frame)
        if self._frame_channel == 1:
            frame_rect = frame_rect[..., 0]
        return frame_rect
