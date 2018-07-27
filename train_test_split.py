import os
from random import shuffle
from math import floor
from shutil import copyfile


def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.abspath(datadir))
    data_files = list(filter(lambda file: file.endswith('.jpg'), all_files))
    return data_files


def randomize_files(file_list):
    shuffle(file_list)
    return file_list


def get_training_and_testing_sets(file_list):
    split = 0.7
    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing


def create_folders(trainig, testing):
    for file in trainig:
        copyfile(path + file, path + 'train/' + file)

    for file in testing:
        copyfile(path + file, path + 'test/' + file)


path = './data_grouped/forest/'
files = get_file_list_from_dir(path)
files = randomize_files(files)
train, test = get_training_and_testing_sets(files)
create_folders(train, test)
