# adapted from https://github.com/danielmurray/adaptiv/

import collections
import numpy as np

from utils import np_utils


class StepDecider:
    # Average seconds per step
    # 0.4 seconds
    PACE = 0.4
    PACE_BUFFER_MAX = 20

    # Average step jerk
    # 2.5 m/s**3
    JERK = 2.5
    JERK_BUFFER_MAX = 20

    def __init__(self, pace_buffer_max, jerk_buffer_max):
        self.jerk_buffer = collections.deque(maxlen=jerk_buffer_max)
        self.jerk_buffer.append(StepDecider.JERK)

        self.pace_buffer = collections.deque(maxlen=pace_buffer_max)
        self.pace_buffer.append(StepDecider.PACE)

        self.last_peak = None
        self.last_trough = None

        # Graphing Purpose Array
        # 0 - timestamp
        # 1 - Jerk Average
        # 2 - Pace Duration
        self.avgs = []

    def decide(self, peak, trough):
        # Given a peak and a trough, determine if the jerk spike
        # and pace spacing is a step

        jerk_avg = sum(self.jerk_buffer) / len(self.jerk_buffer)
        pace_avg = sum(self.pace_buffer) / len(self.pace_buffer)

        jerk = peak['val'] - trough['val']
        pace = abs(peak['ts'] - trough['ts'])

        if self.last_peak and self.last_trough:
            peak_pace = peak['ts'] - self.last_peak['ts']
            trough_pace = trough['ts'] - self.last_trough['ts']
            # print(peak_pace, trough_pace)

            # print('peak', self.last_peak['ts'], peak['ts'], peak_pace)
            # print('trough', self.last_trough['ts'], trough['ts'], trough_pace)

            pace = max(peak_pace, trough_pace)
        else:
            pace = pace_avg

        self.last_peak = peak
        self.last_trough = trough

        self.avgs.append([
            max(peak['ts'], trough['ts']),
            jerk_avg,
            float(pace_avg) / 10 ** 8,
        ])

        # print('')
        # print('jerk', jerk, jerk_avg, jerk > jerk_avg * .5)
        if jerk >= jerk_avg * .5 or jerk >= StepDecider.JERK * 2:
            # print('pace {0:.2f} < {1:.2f} < {2:.2f} -- {3}'.format(0.5 * pace_avg, pace, 2.0 * pace_avg, pace_avg * .5 <= pace <= pace_avg * 2))
            # print('pace', float(pace), float(pace_avg), pace >= pace_avg * .5, pace <= pace_avg * 2, pace >= pace_avg * .5 and pace <= pace_avg * 2)
            if pace_avg * .5 <= pace <= pace_avg * 2:
                self.jerk_buffer.append(jerk)
                self.pace_buffer.append(pace)

                return True
            else:
                return False
        else:
            return False

    def get_avgs(self):
        return self.avgs


class Pedometer(object):

    def __init__(self, acc, times):
        self._acc = np.array(acc)
        self._times = np.array(times)
        self._hz = 1. / np.convolve(self._times, [1, -1], mode='valid').mean()

    def _adaptive_jerk_pace_buffer(self, data, timestamps):
        last_peak = None
        last_trough = None
        last_datum = None
        last_slope = None

        peaks = []
        troughs = []

        sd = StepDecider(StepDecider.PACE_BUFFER_MAX, StepDecider.JERK_BUFFER_MAX)

        for i, datum in enumerate(data):

            timestamp = timestamps[i]

            if last_datum:
                if datum > last_datum:
                    slope = 'rising'
                elif datum < last_datum:
                    slope = 'falling'

                if last_slope and last_slope is not slope:

                    if slope is 'falling':
                        # Maximum
                        potential_peak = {
                            "ts": float(timestamp),
                            "val": float(datum),
                            "index": i,
                            "min_max": "max"
                        }

                        if last_trough:
                            # print('trough?')
                            if sd.decide(potential_peak, last_trough):
                                # print('trough added')
                                troughs.append(last_trough)
                                # last_peak = potential_peak
                        # 	elif last_peak is None or  potential_peak['val'] > last_peak['val']:
                        # 		last_peak = potential_peak
                        # else:
                        last_peak = potential_peak

                    if slope is 'rising':
                        # Minimum
                        potential_trough = {
                            "ts": float(timestamp),
                            "val": float(datum),
                            "index": i,
                            "min_max": "min"
                        }

                        if last_peak:
                            # print('peak?')
                            if sd.decide(last_peak, potential_trough):
                                # print('peak added')
                                peaks.append(last_peak)
                                # last_trough = potential_trough
                        # 	elif last_trough is None or potential_trough['val'] < last_trough['val']:
                        # 		last_trough = potential_trough
                        # else:
                        last_trough = potential_trough

                last_slope = slope
            last_datum = datum
        # print(i)

        return np.array(peaks), np.array(troughs), np.array(sd.avgs)

    def get_step_times(self):
        acc_mag = np.linalg.norm(self._acc, axis=1)
        acc_mag_filt = np_utils.butter_lowpass_filter(acc_mag, cutoff=2., fs=200., order=3)
        jumps, troughs, avgs = self._adaptive_jerk_pace_buffer(acc_mag_filt, self._times)
        step_times = np.array([j['ts'] for j in jumps])
        return step_times

    def get_step_rates(self):
        step_times = self.get_step_times()

        # split the 'mass' of each step among the acc times
        # NOTE(greg): need this b/c the slower you run --> less data --> naively would miss stops
        window_size = int(1. * self._hz)
        step_dts_unmassed = np.zeros(len(self._times), dtype=np.float32)
        for i in range(len(step_times) - 1):
            idxs = np.logical_and(self._times >= step_times[i], self._times < step_times[i+1])
            step_dts_unmassed[idxs] = (step_times[i+1] - step_times[i]) / float(window_size)
        first_idx = np.where(self._times >= step_times[0])[0][0]
        last_idx = np.where(self._times < step_times[-1])[0][-1]

        # sum over the window size to get a smooth signal
        step_dts = np.convolve(step_dts_unmassed, np.ones(window_size), mode='same')
        step_dts = np.clip(step_dts, 1e-4, np.inf)
        step_rates = 1. / step_dts
        step_rates[:first_idx+window_size] = step_rates[first_idx+window_size] + \
            np.random.normal(scale=0.01, size=first_idx+window_size)
        step_rates[last_idx-window_size:] = step_rates[last_idx-window_size] + \
            np.random.normal(scale=0.01, size=len(step_rates) - last_idx + window_size)
        step_rates = np.clip(step_rates, 0, np.inf)

        too_fast = step_rates > 3.5
        if np.any(too_fast):
            print('{0:.1f} ({1}%) seconds is too fast!!!'.format(
                too_fast.sum() / float(self._hz),
                int(100 * self._times[too_fast].sum() / self._times.sum())
            ))
        step_rates = np.clip(step_rates, 0., 3.5)

        return step_rates
