import cv2
import h5py
import numpy as np
import os
import pandas

from gopro.equirectangular_to_dualfisheye import EquirectangularToDualfisheye
from utils import file_utils, np_utils, transformations
from gopro.pedometer import Pedometer
from utils.python_utils import AttrDict


class GoproToHdf5(object):

    FPS = 3

    def __init__(self, folder):
        assert os.path.exists(folder)
        self._hdf5_folder = folder

        # files
        csv_acc_fnames = file_utils.get_files_ending_with(folder, 'ACCL.csv')
        assert len(csv_acc_fnames) > 0
        self._csv_acc_fname = csv_acc_fnames[0]

        csv_ori_fnames = file_utils.get_files_ending_with(folder, 'CORI.csv')
        assert len(csv_ori_fnames) > 0
        self._csv_ori_fname = csv_ori_fnames[0]

        csv_gyro_fnames = file_utils.get_files_ending_with(folder, 'GYRO.csv')
        assert len(csv_gyro_fnames) > 0
        self._csv_gyro_fname = csv_gyro_fnames[0]

        csv_mag_fnames = file_utils.get_files_ending_with(folder, 'MAGN.csv')
        assert len(csv_mag_fnames) > 0
        self._csv_mag_fname = csv_mag_fnames[0]

        mp4_fnames = file_utils.get_files_ending_with(folder, '.mp4')
        assert len(mp4_fnames) == 1
        self._mp4_fname = mp4_fnames[0]

        # mp4
        self._cap = cv2.VideoCapture(self._mp4_fname)
        self._mp4_fps = int(np.round(self._cap.get(cv2.CAP_PROP_FPS)))
        assert self._mp4_fps % GoproToHdf5.FPS == 0
        print('MP4 file: {0}'.format(self._mp4_fname))
        print('MP4 fps: {0}'.format(self._mp4_fps))
        self._equi_to_dualfisheye = EquirectangularToDualfisheye()

        ### iterating
        # image
        self._num_frames = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self._curr_frame = None
        self._frame_exists = False
        self._frame_num = 0
        self._frame_skips = 0
        self._frame_time = 0.
        # data
        self._frames_since_last_save = np.inf
        self._save_every_nth = self._mp4_fps // GoproToHdf5.FPS
        self._hdf5_dict = AttrDict()
        self._hdf5_dict.time = []
        self._hdf5_dict.image = []
        self._hdf5_basename = folder.strip('/').split('/')[-1]

        print('Init complete')

    ###########
    ### Run ###
    ###########

    @property
    def _mp4_is_done(self):
        return not self._cap.isOpened() or self._frame_num >= self._num_frames

    def _mp4_step(self):
        self._frame_exists, read_frame = self._cap.read()
        self._frame_time = 1e-3 * self._cap.get(cv2.CAP_PROP_POS_MSEC)
        self._frame_num += 1
        if self._frame_exists:
            self._frame_skips = 0
            self._curr_frame = read_frame
        else:
            self._frame_skips += 1

    def _hdf5_step(self):
        if self._frames_since_last_save >= self._save_every_nth:
            self._frames_since_last_save = 0

            self._hdf5_dict.time.append(self._frame_time)

            curr_frame = cv2.cvtColor(self._curr_frame, cv2.COLOR_BGR2RGB)
            curr_frame = self._equi_to_dualfisheye.to_rectified(curr_frame)
            curr_frame = np_utils.imresize(curr_frame, [90, 160, 3])
            self._hdf5_dict.image.append(curr_frame)

        self._frames_since_last_save += 1

    def _hdf5_add_imu(self):
        ### read csv files

        hdf5_times = list(self._hdf5_dict.time)
        hdf5_times.append(hdf5_times[-1] + (hdf5_times[-1] - hdf5_times[-2]))
        for name, fname, parse_func in [
            ('imu/accelerometer', self._csv_acc_fname,
             lambda csv: [csv['Accelerometer (x) [m/s2]'], csv['Accelerometer (y) [m/s2]'], csv['Accelerometer (z) [m/s2]']]),
            ('imu/gyroscope', self._csv_gyro_fname,
             lambda csv: [csv['Gyroscope (x) [rad/s]'], csv['Gyroscope (y) [rad/s]'], csv['Gyroscope (z) [rad/s]']]),
            ('imu/quaternion', self._csv_ori_fname,
             lambda csv: [csv['CameraOrientation'], csv['1'], csv['2'], csv['3']]),
            ('imu/magnetometer', self._csv_mag_fname,
             lambda csv: [csv['1'], csv['2'], csv['3']])]:
            # read
            csv = pandas.read_csv(fname)
            times = 1e-3 * np.array(csv['cts'])
            values = np.stack(parse_func(csv)).T.astype(np.float32)
            if 'quaternion' in name:
                values /= np.linalg.norm(values, axis=1)[:, np.newaxis]

            # get rid of values outside of time
            valids = np.logical_and(times >= hdf5_times[0], times < hdf5_times[-1])
            times = times[valids]
            values = values[valids]

            # chunk
            indices = np.digitize(times, hdf5_times) - 1
            values_chunked = [values[indices == i] for i in range(len(hdf5_times) - 1)]
            self._hdf5_dict[name] = values_chunked

    def _hdf5_add_commands(self):
        ### all
        hdf5_times = list(self._hdf5_dict.time)
        hdf5_times.append(hdf5_times[-1] + (hdf5_times[-1] - hdf5_times[-2]))


        ### steps
        csv = pandas.read_csv(self._csv_acc_fname)
        times = 1e-3 * np.array(csv['cts'])
        accs = np.stack([csv['Accelerometer (x) [m/s2]'],
                         csv['Accelerometer (y) [m/s2]'],
                         csv['Accelerometer (z) [m/s2]']]).T.astype(np.float32)

        pedometer = Pedometer(accs, times)
        step_rates = pedometer.get_step_rates()

        step_rates_chunked = []
        for i in range(len(hdf5_times) - 1):
            indices = np.logical_and(times >= hdf5_times[i], times < hdf5_times[i+1])
            step_rate_i = step_rates[indices].mean()
            step_rates_chunked.append(step_rate_i)


        ### camera orientation
        csv = pandas.read_csv(self._csv_ori_fname)
        times = np.array(1e-3 * csv['cts'])
        quats = np.array([csv['1'], csv['2'], csv['3'], csv['CameraOrientation']]).T

        def smooth_quaternions(quats, frac_range=0.4, frac_bias=0.04):
            # https://www.mathworks.com/help/fusion/examples/lowpass-filter-orientation-using-quaternion-slerp.html
            smoothed_quats = [quats[0]]
            for quat in quats[1:]:
                smoothed_quat = smoothed_quats[-1]
                delta_quat = transformations.quaternion_multiply(transformations.quaternion_inverse(smoothed_quat), quat)
                delta_quat /= np.linalg.norm(delta_quat)
                angle = transformations.quaternion_to_axisangle(delta_quat)[0][0]
                alpha = (angle / np.pi) * frac_range + frac_bias
                smoothed_quat = transformations.quaternion_slerp(smoothed_quats[-1], quat, alpha)
                smoothed_quats.append(smoothed_quat)
            smoothed_quats = np.array(smoothed_quats)
            assert np.all(np.isfinite(smoothed_quats))
            return smoothed_quats
        quats = smooth_quaternions(quats)

        quats_chunked = []
        for i in range(len(hdf5_times) - 1):
            indices = np.logical_and(times >= hdf5_times[i], times < hdf5_times[i+1])
            quats_chunked.append(quats[indices])


        ### integrate
        dt = times[1] - times[0]
        fps = 1. / dt
        turns = []
        steps = []
        for i in range(len(hdf5_times) - 1):
            indices = np.where(np.logical_and(times >= hdf5_times[i], times < hdf5_times[i + 1]))[0]
            assert len(indices) == (indices[-1] - indices[0] + 1)
            start_idx, stop_idx = indices[0], indices[-1]

            # step
            step_rate = step_rates_chunked[i]

            # angles
            prepend = int(fps * 1.5)
            prestart_idx = np.clip(start_idx - prepend, 0, len(times) - 1)
            poststop_idx = np.clip(stop_idx + prepend, 0, len(times) - 1)

            origin = quats[start_idx]
            origin_inv = transformations.quaternion_inverse(origin)
            quats_list_origin = [transformations.quaternion_multiply(origin_inv, q) for q in quats[prestart_idx:poststop_idx]]
            angles = -np.array([transformations.euler_from_quaternion(q)[1] for q in quats_list_origin])
            angles = np_utils.butter_lowpass_filter(angles, cutoff=0.8, fs=30., order=5)

            # account for delay of
            delay_shift = int(0.8 * fps)
            angles = np.roll(angles, -delay_shift)

            angles = angles[start_idx-prestart_idx:-(poststop_idx-stop_idx)]

            pos = np.zeros(2)
            if len(angles) > 0:
                dt = (1. / GoproToHdf5.FPS) / len(angles)
                for angle in angles:
                    pos += dt * step_rate * np.array([np.cos(angle), np.sin(angle)])

            turns.append(np.arctan2(pos[1], pos[0]))
            steps.append(np.linalg.norm(pos))

        self._hdf5_dict.commands.step = np.array(steps)
        self._hdf5_dict.commands.turn = np.array(turns)

    def _hdf5_save(self):
        hdf5_fname = os.path.join(self._hdf5_folder, '{0:03d}.hdf5'.format(0))
        with h5py.File(hdf5_fname, 'w') as f:
            f['image'] = np.array(np_utils.im2bytes(np.array(self._hdf5_dict.image)))
            f['step'] = np.array(self._hdf5_dict.commands.step)
            f['turn'] = np.array(self._hdf5_dict.commands.turn)

    def _close(self):
        self._cap.release()

    def run(self):
        while not self._mp4_is_done:

            if self._frame_num % 100 == 0:
                print('Frame number: {0}'.format(self._frame_num))

            self._mp4_step()
            self._hdf5_step()

        self._hdf5_add_imu()
        self._hdf5_add_commands()
        self._hdf5_save()
        self._close()
