import os
import shutil
from tqdm import tqdm

def clear_folder(folder_path):
    for file in os.listdir(folder_path):
        if not file.endswith('.gitkeep'):
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

    subdir_paths = [os.path.join(folder_path, subdir) for subdir in os.listdir(folder_path)]
    for subdir_path in tqdm(subdir_paths, desc = f'Cleaning up {folder_path}'):
        if os.path.isdir(subdir_path):
            if len(os.listdir(subdir_path)) == 0:
                os.rmdir(subdir_path)
