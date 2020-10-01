import argparse
from multiprocessing import Pool
import os

from gopro.gopro_to_hdf5 import GoproToHdf5
from gopro.hdf5_visualizer import HDF5Visualizer
from utils import file_utils

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='command')

gopro_to_hdf5_parser = subparsers.add_parser('gopro_to_hdf5')
gopro_to_hdf5_parser.add_argument('folder', type=str, help='find gopro folders in subdirectories')
gopro_to_hdf5_parser.add_argument('--threads', type=int, default=1)

hdf5_visualizer_parser = subparsers.add_parser('hdf5_visualizer')
hdf5_visualizer_parser.add_argument('folder', type=str, help='find hdf5 files in subdirectories')
hdf5_visualizer_parser.add_argument('-horizon', type=int, default=15)

args = parser.parse_args()

if args.command == 'gopro_to_hdf5':
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
        pool.map(run, folders)
elif args.command == 'hdf5_visualizer':
    hdf5_fnames = sorted(file_utils.recursive_get_files_ending_with(args.folder, '.hdf5'))
    assert len(hdf5_fnames) > 0
    for fname in hdf5_fnames:
        print(fname)
    HDF5Visualizer(hdf5_fnames, args.horizon).run()
else:
    raise ValueError
