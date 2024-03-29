"""Util functions for operations like deleting a folder, etc."""

import os
import shutil
from tqdm import tqdm
from functools import partial


def listdir(path):
    """Modified os.listdir to keep out hidden files."""
    return [f for f in os.listdir(path) if not f.startswith('.')]


def clear_folder(folder_path):
    """Deletes all files in given folder path."""
    for file in listdir(folder_path):
        os.remove(os.path.join(folder_path, file))


def clear_subfolders_in_folder(folder_path):
    """Deletes subfolders in given folder path."""
    for subfolder in listdir(folder_path):
        clear_folder(os.path.join(folder_path, subfolder))
        os.rmdir(os.path.join(folder_path, subfolder))


def make_folder(target_folder):
    """Makes the target folder (empty)."""
    if os.path.exists(target_folder):
        clear_folder(target_folder)
    else:
        os.mkdir(target_folder)


def copy_tree(source_folder, destination_folder):
    """Copies source folder to destination folder."""
    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)
    shutil.copytree(source_folder, destination_folder)


def get_empty_subdirectories(folder_path):
    """
    Function to get empty and non empty subdirectories in folder path.

    Where is it used in the codebase?
    Dataset classes search for video subfolders (with frames) in a folder, in their
    __init__ function. Video subfolders that are empty are filtered using this function.
    """
    empty_subdir_paths = []
    non_empty_subdir_paths = []
    subdir_paths = [os.path.join(folder_path, subdir) for subdir in listdir(folder_path)]

    for subdir_path in tqdm(subdir_paths, desc = f'Examining {folder_path}'):
        if os.path.isdir(subdir_path):
            if len(listdir(subdir_path)) == 0:
                empty_subdir_paths.append(subdir_path)
            else:
                non_empty_subdir_paths.append(subdir_path)

    return empty_subdir_paths, non_empty_subdir_paths


def clear_empty_subdirectories(folder_path):
    """
    Function to delete empty subdirectories in folder path.

    Where is it used in the codebase?
    It is not used in the codebase. This function has been replaced by the less strict
    get_empty_subdirectories() function that does not delete empty subdirectories but
    simply returns their paths and they are handled elsewhere.
    """
    num_cleared = 0
    subdir_paths = [os.path.join(folder_path, subdir) for subdir in listdir(folder_path)]
    for subdir_path in tqdm(subdir_paths, desc = f'Cleaning up {folder_path}'):
        if os.path.isdir(subdir_path):
            if len(listdir(subdir_path)) == 0:
                os.rmdir(subdir_path)
                num_cleared += 1
    return num_cleared


def create_dataset_structure(root_directory, val_required = False):
    """Create standard binary classification setting dataset structure."""

    folders_list = ['train', 'test', 'train/1', 'train/0', 'test/1', 'test/0']

    if val_required:
        folders_list.extend(['val', 'val/1', 'val/0'])

    # https://www.geeksforgeeks.org/make-multiple-directories-based-on-a-list-using-python/
    concat_root_path = partial(os.path.join, root_directory)
    make_directory = partial(os.makedirs, exist_ok=True)

    for path_items in map(concat_root_path, folders_list):
        make_directory(path_items)


def create_class_structure(root_directory):
    """Create standard binary classification setting class structure."""

    folders_list = ['1', '0']
    concat_root_path = partial(os.path.join, root_directory)
    make_directory = partial(os.makedirs, exist_ok=True)

    for path_items in map(concat_root_path, folders_list):
        make_directory(path_items)
