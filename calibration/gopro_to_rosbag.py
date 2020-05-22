import argparse
import cv2
import os

import rosbag
import rospy
import ros_numpy
from sensor_msgs.msg import Image

from gopro import equirectangular_to_dualfisheye
from utils import file_utils


class StampAndMessage(object):

    def __init__(self, name, stamp, message):
        self.name = name
        self.stamp = stamp
        self.message = message

    def __lt__(self, other):
        return self.stamp < other.stamp


class GoproToRosbag(object):

    def __init__(self, folder):
        assert os.path.exists(folder)

        mp4_fnames = file_utils.get_files_ending_with(folder, '.mp4')
        assert len(mp4_fnames) == 1
        self._mp4_fname = mp4_fnames[0]

        self._remap = None
        self._keep_mask = None

    ##############
    ### Camera ###
    ##############

    def _image_iterator(self):
        full_height, full_width = 1920, 3840

        cap = cv2.VideoCapture(self._mp4_fname)
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_num = 0
        seq = 0
        while cap.isOpened() and (frame_num < num_frames):
            frame_exists, curr_frame = cap.read()
            frame_num += 1
            if frame_exists and frame_num % 10 == 0:
                stamp = rospy.Time.from_sec(1e-3 * cap.get(cv2.CAP_PROP_POS_MSEC))

                curr_frame, self._remap, self._keep_mask = \
                    equirectangular_to_dualfisheye.equirectangular_to_dualfisheye(curr_frame,
                                                                                  [1000, 2000],
                                                                                  remap=self._remap,
                                                                                  keep_mask=self._keep_mask)
                curr_frame = curr_frame[:, 1000:]

                curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                encoding = 'mono8'

                # assert tuple(curr_frame.shape[:2]) == (full_height, full_width)

                # image_msg = CompressedImage()
                # image_msg.header.stamp = stamp
                # image_msg.format = 'jpeg'
                # image_msg.data = np.array(cv2.imencode('.jpg', curr_frame)[1]).tostring()

                image_msg = ros_numpy.msgify(Image, curr_frame, encoding=encoding)
                image_msg.header.stamp = stamp
                image_msg.header.seq = seq

                yield image_msg

                seq += 1

        cap.release()

    def run(self):
        ### iterators
        print('Creating iterators...')
        image_iter = self._image_iterator()

        image_msg = next(image_iter)

        print('Writing rosbag...')
        bag_fname = self._mp4_fname.replace('.mp4', '')
        bag_fname += '.bag'
        bag = rosbag.Bag(bag_fname, 'w')
        try:
            while True:
                bag.write('cam/image/raw', image_msg, t=image_msg.header.stamp)
                image_msg = next(image_iter)
        except StopIteration:
            pass
        bag.close()


parser = argparse.ArgumentParser()
parser.add_argument('folder', type=str)
args = parser.parse_args()

GoproToRosbag(args.folder).run()
