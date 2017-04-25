import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sklearn
import os
import pickle

script_dir = os.path.dirname(__file__)
print(script_dir)
train_dir = script_dir + '/CIKM2017_train'
train_file = 'train.txt'


def read_data(test=True):
    data = []
    with open(os.path.join(train_dir, train_file)) as file:
        print('processing start')
        index = 0
        for line in file:
            if test and index > 5:
                break
            data.append(line.split(','))
            index += 1
    # print('processing..:{}'.format(index))
    print('processing done!')

    return data


def save_data_to_pickle(data, test=True):
    keys = ['no', 'label', 'radar_maps']

    pickle_dir = 'pickle'
    if test:
        pickle_dir += '_test'

    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)

    radar_maps_stddev, radar_maps_mean = 0, 0

    for index, datum in enumerate(data):
        pickle_data = dict(zip(keys, parse_single_data(datum)))
        file_name = pickle_data['no']

        radar_maps_mean += np.mean(pickle_data['radar_maps'])
        radar_maps_stddev += np.std(pickle_data['radar_maps'])

        if index % 500 == 0:
            print('saving .. {}'.format(index))

        try:
            with open(os.path.join(pickle_dir, file_name+'.pickle'), 'wb') as f:
                pickle.dump(pickle_data, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('pickle except')
            print(e)

    print('mean == {}'.format(radar_maps_mean/len(data)))
    print('std == {}'.format(radar_maps_stddev/len(data)))

    # mean == 48.446656059536686
    # std == 41.04652582906467
    return True


def parse_single_data(single_data):
    '''

    :param single_data:  ('train_no', 'label', 'HTS(15*4*101*101)')
    :return: train_no, label: float, radar_maps: list
    '''

    train_no, labels, radar_maps = single_data
    labels = float(labels)

    radar_maps = np.array(list(map(float, radar_maps.split())))

    TIME_POINT, HEIGHT = 15, 4
    radar_maps = np.reshape(radar_maps, newshape=(TIME_POINT, HEIGHT, 101, 101))

    return train_no, labels, radar_maps


data = read_data()
assert data[0] is not None
single = data[0]
assert isinstance(parse_single_data(single)[0], str)
assert parse_single_data(single)[2].shape == (15, 4, 101, 101)

assert save_data_to_pickle(data)

print('test done!')

if __name__ == '__main__':
    save_data_to_pickle(read_data(test=False), test=False)
    print('save done!')
