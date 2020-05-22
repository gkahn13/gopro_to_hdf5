import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from utils.gopro_utils import turns_and_steps_to_positions
from utils import file_utils, np_utils, pyblit
from utils.python_utils import Getch


class HDF5Visualizer(object):

    def __init__(self, data_fnames, horizon):
        for fname in data_fnames:
            assert os.path.exists(fname)

        self._data_fnames = data_fnames

        self._curr_data_idx = 0
        self._prev_data_idx = -1
        self._curr_data_timestep = 0
        self._curr_data = None
        self._curr_data_len = None
        self._load_data()

        self._horizon = horizon
        self._setup_visualization()

    def _setup_visualization(self):
        self._f, axes = plt.subplots(1, 4, figsize=(20, 3))
        ax_im, ax_poses, ax_turn, ax_step = axes
        self._plot_is_showing = False

        self._pyblit_im = pyblit.Imshow(ax_im)
        self._pyblit_im_ax = pyblit.Axis(ax_im, [self._pyblit_im])

        self._pyblit_poses = pyblit.Line(ax_poses)
        self._pyblit_poses_ax = pyblit.Axis(ax_poses, [self._pyblit_poses])

        self._pyblit_turn_bar = pyblit.Barh(ax_turn)
        self._pyblit_turn_ax = pyblit.Axis(ax_turn, [self._pyblit_turn_bar])

        self._pyblit_step_bar = pyblit.Bar(ax_step)
        self._pyblit_step_ax = pyblit.Axis(ax_step, [self._pyblit_step_bar])

    def get(self, key, horizon=None):
        horizon = horizon or self._horizon
        start = self._curr_data_timestep
        end = min(start + horizon, self._curr_data_len)
        value = self._curr_data[key][start:end]
        if key == 'image':
            value = np_utils.bytes2im(value)
        return value

    def _load_data(self):
        if self._curr_data is not None:
            self._curr_data.close()
        self._curr_data = h5py.File(self._data_fnames[self._curr_data_idx], 'r')
        self._curr_data_len = len(self._curr_data['turn']) - 1

    @property
    def no_more_timesteps(self):
        return self._curr_data_timestep >= self._curr_data_len - 1

    @property
    def no_more_files(self):
        return self._curr_data_idx == len(self._data_fnames) - 1

    def next_timestep(self):
        if self.no_more_timesteps and self.no_more_files:
            return # at the end, do nothing

        self._curr_data_timestep += 1
        if self.no_more_timesteps:
            self._curr_data_idx += 1
            self._load_data()
            self._curr_data_timestep = 0

    def prev_timestep(self):
        if (self._curr_data_timestep == 0) and (self._curr_data_idx == 0):
            return # at the beginning, do nothing

        self._curr_data_timestep -= 1
        if self._curr_data_timestep < 0:
            self._curr_data_idx -= 1
            self._load_data()
            self._curr_data_timestep = self._curr_data_len - 2

    def next_data(self):
        if self.no_more_files:
            return # at the end, do nothing

        self._curr_data_idx += 1
        self._load_data()
        self._curr_data_timestep = 0

    def prev_data(self):
        if self._curr_data_idx == 0:
            return # at the beginning, do nothing

        self._curr_data_idx -= 1
        self._load_data()
        self._curr_data_timestep = 0

    def next_data_end(self):
        if self.no_more_files:
            pass # at the end, do nothing
        else:
            self._curr_data_idx += 1

        self._load_data()
        self._curr_data_timestep = self._curr_data_len - 2

    def prev_data_end(self):
        if self._curr_data_idx == 0:
            pass  # at the end, do nothing
        else:
            self._curr_data_idx -= 1

        self._load_data()
        self._curr_data_timestep = self._curr_data_len - 2


    #################
    ### Visualize ###
    #################

    def _plot_im(self, pyblit_im, pyblit_im_ax, topic):
        images = self.get(topic, horizon=1)
        images = images[::-1].reshape([-1] + list(images.shape[2:]))
        pyblit_im.draw(images)
        pyblit_im_ax.draw()

    def _plot_positions(self):
        turns = self.get('turn')
        steps = self.get('step')

        positions = turns_and_steps_to_positions(turns, steps)
        self._pyblit_poses.draw(positions[:, 1], positions[:, 0])

        ax = self._pyblit_poses_ax.ax
        max_speed = 3.5
        dt = 0.3
        max_position = max_speed * dt * self._horizon
        ax.set_xlim((-max_position, max_position))
        ax.set_ylim((-0.1, max_position))
        ax.set_aspect('equal')

        self._pyblit_poses_ax.draw()

    def _plot_turn(self):
        turns = self.get('turn')
        self._pyblit_turn_bar.draw(np.arange(len(turns)), turns)
        ax = self._pyblit_turn_ax.ax
        ax.set_xlim((-0.5, 0.5))
        ax.set_xlabel('turn')
        ax.set_ylabel('timestep')
        ax.set_yticks(np.arange(len(turns)))
        self._pyblit_turn_ax.draw()

    def _plot_step(self):
        steps = self.get('step')
        self._pyblit_step_bar.draw(np.arange(len(steps)), steps)
        ax = self._pyblit_step_ax.ax
        ax.set_ylim((0., 1.5))
        ax.set_ylabel('step')
        ax.set_xlabel('timestep')
        ax.set_xticks(np.arange(len(steps)))
        self._pyblit_step_ax.draw()

    def _update_visualization(self):
        self._plot_im(self._pyblit_im, self._pyblit_im_ax, 'image')
        self._plot_positions()
        self._plot_turn()
        self._plot_step()

        if not self._plot_is_showing:
            plt.show(block=False)
            self._plot_is_showing = True
            plt.pause(0.01)
            # self._f.tight_layout()
        self._f.canvas.flush_events()

    ###########
    ### Run ###
    ###########

    def run(self):
        self._update_visualization()

        while True:
            print('{0}/{1}, {2}/{3} -- {4}'.format(self._curr_data_timestep+1, self._curr_data_len,
                                                   self._curr_data_idx+1, len(self._data_fnames),
                                                   os.path.basename(self._data_fnames[self._curr_data_idx])))
            self._update_visualization()

            char = Getch.getch()
            self._prev_data_idx = self._curr_data_idx

            if char == 'q':
                break
            elif char == 'e':
                self.next_timestep()
            elif char == 'w':
                self.prev_timestep()
            elif char == 'd':
                self.next_data()
            elif char == 's':
                self.prev_data()
            elif char >= '1' and char <= '9':
                for _ in range(int(char)):
                    self.next_timestep()
            elif char == 'c':
                self.next_data_end()
            elif char == 'x':
                self.prev_data_end()
            else:
                continue

