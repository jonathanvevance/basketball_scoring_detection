import os
import shutil
from tqdm import tqdm
from functools import partial


def listdir(path):
    return [f for f in listdir(path) if not f.startswith('.')]


def clear_folder(folder_path):
    for file in listdir(folder_path):
        os.remove(os.path.join(folder_path, file))


def make_folder(target_folder):
    if os.path.exists(target_folder):
        clear_folder(target_folder)
    else:
        os.mkdir(target_folder)


def copy_tree(source_folder, destination_folder):
    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)
    shutil.copytree(source_folder, destination_folder)


def clear_empty_subdirectories(folder_path):

    subdir_paths = [os.path.join(folder_path, subdir) for subdir in listdir(folder_path)]
    for subdir_path in tqdm(subdir_paths, desc = f'Cleaning up {folder_path}'):
        if os.path.isdir(subdir_path):
            if len(listdir(subdir_path)) == 0:
                os.rmdir(subdir_path)


def create_dataset_structure(root_directory, val_required = False):

    folders_list = ['train', 'test', 'train/1', 'train/0', 'test/1', 'test/0']

    if val_required:
        folders_list.extend(['val', 'val/1', 'val/0'])

    # https://www.geeksforgeeks.org/make-multiple-directories-based-on-a-list-using-python/
    concat_root_path = partial(os.path.join, root_directory)
    make_directory = partial(os.makedirs, exist_ok=True)

    for path_items in map(concat_root_path, folders_list):
        make_directory(path_items)


def create_class_structure(root_directory):

    folders_list = ['1', '0']
    concat_root_path = partial(os.path.join, root_directory)
    make_directory = partial(os.makedirs, exist_ok=True)

    for path_items in map(concat_root_path, folders_list):
        make_directory(path_items)
