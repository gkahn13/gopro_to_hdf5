import argparse
from multiprocessing import Pool
import os

from gopro_to_hdf5 import GoproToHdf5
import file_utils


parser = argparse.ArgumentParser()
parser.add_argument('folder', type=str, help='find gopro folders in subdirectories')
parser.add_argument('--threads', type=int, default=1)
args = parser.parse_args()

assert os.path.exists(args.folder)
dot360_fnames = file_utils.recursive_get_files_ending_with(args.folder, '.360')
folders = [os.path.dirname(fname) for fname in dot360_fnames]

if args.threads == 1:
    for folder in folders:
        GoproToHdf5(folder).run()
else:
    def run(f):
        GoproToHdf5(f).run()
    pool = Pool(args.threads)
    pool.map(run, args.folders)
