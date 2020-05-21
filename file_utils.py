import h5py
import importlib.util
import itertools
import os
from pathlib import Path


def get_files_ending_with(folder_or_folders, ext):
    if isinstance(folder_or_folders, str):
        folder = folder_or_folders
        assert os.path.exists(folder)

        fnames = []
        for fname in os.listdir(folder):
            if fname.endswith(ext):
                fnames.append(os.path.join(folder, fname))
        return sorted(fnames)
    else:
        assert hasattr(folder_or_folders, '__iter__')
        return sorted(list(itertools.chain(*[get_files_ending_with(folder, ext) for folder in folder_or_folders])))


def recursive_get_files_ending_with(folder_or_folders, ext):
    if isinstance(folder_or_folders, str):
        return sorted([str(path) for path in Path(folder_or_folders).rglob('*{0}'.format(ext))])
    else:
        assert hasattr(folder_or_folders, '__iter__')
        return sorted(list(itertools.chain(*[recursive_get_files_ending_with(folder, ext)
                                             for folder in folder_or_folders])))
