'''
Normalize radar map to mean == 0, and stddev = 0
'''

import numpy as np
import pickle
import os


def normalize_single_one(radar_maps):
    MEAN = 48.446656059536686
    STD = 41.04652582906467
    radar_maps = (radar_maps - MEAN) / STD
    return radar_maps


def normalize_batch(train_size=10000, dir='pickle'):
    for i in range(train_size):

        if i % 100 == 0:
            print('normalizing: {}..'.format(i))

        file_name = os.path.join(dir, 'train_{}.pickle'.format(i))

        try:
            new_data = process_single(file_name)

            with open(file_name, 'wb') as f:
                pickle.dump(new_data, f)
        except FileNotFoundError as fe:
            print('file not exist: {}'.format(file_name))

    print('done!')


def process_single(file_name):

    with open(file_name, 'rb') as f:
        datum = pickle.load(f)
        datum['radar_maps'] = normalize_single_one(datum['radar_maps'])

    return datum

if __name__ == '__main__':
    for i in range(100):
        index = np.random.choice(range(10000), size=1)[0]
        datum = process_single('./pickle/train_{}.pickle'.format(index))

        assert np.abs(np.mean(datum['radar_maps'])-0.0) < 5.0
        assert np.std(np.mean(datum['radar_maps'])-1.0) < 5.0

    print('test done!')

