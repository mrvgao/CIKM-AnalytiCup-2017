'''
Get a regression model to get the right prediction. 

Version 0.1: Line Regression, treat one label's radar maps as X (4 * 15 * 101 * 101),

Compress X to 101 * 101, in other words, get the mean of radar maps of one label.

    yhat = aXb + c
    (
        yhat.shape = (),
        X.shape = (101, 101)
        a.shape = (1, 101)
        b.shape = (101, 1)
        c.shape = ()
    )
    
    Loss = l2 loss of yhat and y.
    
Author: Minquan Gao
Data: 17-Apr-19
'''

import tensorflow as tf
import numpy as np
import itertools
import evalution
import pickle
import os
import draw_performance
import time

import matplotlib.pyplot as plt

EMOJIS = ['\U0001f601', '\U0001f602', '\U0001f603', '\U0001f604']


class Config:
    learning_rate = 1e-4 # learning_rate
    regularization_rate = 1e-2 # regularization rate
    batch_size = 256
    epoch = 5000
    crop_center = 10
    location = 101

    TIME = 15
    HEIGHT = 4

    train_data_size = 1000
    drop_out = 0.8

    hidden_size = [10, 10, 10, 10, 10, 10, 10]

    matrix_keep_prob = 0.01

    n_steps = 72
    n_hidden = 200

np.random.seed(2)


class RainRegression:
    def __init__(self, test=True):
        self.config = Config()

        self.X_dimension = int(self.config.HEIGHT * self.config.TIME \
                               * self.config.location * self.config.location
                               * self.config.matrix_keep_prob)

        self.parameters = 'parameters'
        self.__add_model()

        self.train_indices, self.validation_indices, self.test_indices = self.split_test_train()

        if test:
            self.config.train_data_size = 128
            self.train_indices = self.train_indices[: self.config.batch_size]
            self.config.epoch = 2

        self.cache = self.__load_data()

        assert len(self.train_indices) / self.config.batch_size >= 1

    def __load_data(self):
        '''
        Loads data in memory to speed up the calculate time.
        :return: 
        '''
        indices = np.concatenate([self.train_indices, self.test_indices, self.validation_indices])

        data_cache = {}

        file_dir = './pickle'
        for i, index in enumerate(indices):
            target_train_file = os.path.join(file_dir, 'train_{}.pickle'.format(index))
            try:
                with open(target_train_file, 'rb') as f:
                    data = pickle.load(f)
                    label = float(data['label'])
                    radar_maps = data['radar_maps']

                    compressed_radar_maps = self.compress_radar_maps(radar_maps)

                    data_cache[index] = (label, compressed_radar_maps)
            except EOFError as e:
                print('{} eof error'.format(target_train_file))
                return None, None
            finally:
                print('.', end='')
                if i % 100 == 0: print('{}/{}'.format(i, len(indices)))

        print('Data Load Finished.')

        return data_cache

    def __add_model(self):
        with tf.variable_scope('train_data') as scope:
            self.X_train = tf.placeholder(tf.float32, shape=(None, self.X_dimension))
            self.labels = tf.placeholder(tf.float32, shape=(None, ))

        # regression = self.MFC(self.X_train)
        # regression = self.RNN(x)
        regression = self.conv_net(self.X_train)

        need_regularize = tf.get_collection(key=self.parameters)

        l2_loss = sum(map(lambda w: tf.nn.l2_loss(w), need_regularize))

        tf.add_to_collection(name='l2_loss', value=l2_loss)

        with tf.variable_scope('get_value') as predict:
            self.yhat = regression

        with tf.variable_scope('loss') as loss_scope:
            self.loss = self.loss(self.yhat)

        with tf.variable_scope('op') as op_scope:
            # op_scope.reuse_variables()
            self.op = self.optimizer(self.loss)

    def RNN(self, x):

        with tf.variable_scope('rnn') as scope:
            x = tf.unstack(x, self.config.n_steps, 1)

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.config.n_hidden, forget_bias=1.0)

            outputs, status = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

            rnn_weight = tf.get_variable('rnn_weight', shape=(self.config.n_hidden, 1),
                                         initializer=tf.truncated_normal_initializer(stddev=0.02))

            rnn_bias = tf.get_variable('rnn_bias', shape=(),
                                       initializer=tf.constant_initializer(0.0))

            tf.add_to_collection(name=self.parameters, value=rnn_weight)
            tf.add_to_collection(name=self.parameters, value=rnn_bias)

            return tf.matmul(outputs[-1], rnn_weight) + rnn_bias

    def MFC(self, input):
        # [fc => relu => batch-normai => ] * 4
        for i in range(len(self.config.hidden_size)):
            fc_layer = 'fc_layer_{}'.format(i)

            with tf.variable_scope(fc_layer) as fc_scope:
                if i == 0:
                    weight_shape = (self.X_dimension, self.config.hidden_size[i])
                    last_layer_output = input
                else:
                    weight_shape = (self.config.hidden_size[i-1], self.config.hidden_size[i])

                bias_shape = (self.config.hidden_size[i], )

                WEIGHT, BIAS = 'weight_{}'.format(i), 'bias_{}'.format(i)

                weights = tf.get_variable(name=WEIGHT, shape=weight_shape,
                                          initializer=tf.truncated_normal_initializer(stddev=0.05))

                bias = tf.get_variable(name=BIAS, shape=bias_shape,
                                       initializer=tf.constant_initializer(0.0))

                tf.add_to_collection(name=self.parameters, value=weights)
                tf.add_to_collection(name=self.parameters, value=bias)

                layer_output = tf.matmul(last_layer_output, weights) + bias
                layer_output = tf.nn.dropout(layer_output, keep_prob=self.config.drop_out)

            with tf.variable_scope('BN') as bn_scope:
                batch_normalized = tf.nn.batch_normalization(layer_output, mean=0, variance=1, offset=0, scale=100,
                                                             variance_epsilon=1e-4)

            with tf.variable_scope('relu_1') as relu_scope:
                relu_output = tf.nn.relu(batch_normalized)
                relu_output = tf.nn.dropout(relu_output, keep_prob=self.config.drop_out)

            last_layer_output = relu_output

        with tf.variable_scope('tanh_1') as tanh:
            tanh_output = tf.tanh(last_layer_output)

        with tf.variable_scope('regression') as regression_scope:
            # shape = (self.X_dimension, 1)
            shape = (self.config.hidden_size[-1], 1)
            a = tf.get_variable('a', shape=shape, initializer=tf.contrib.layers.xavier_initializer(seed=0))

            b = tf.get_variable('b', shape=(), initializer=tf.zeros_initializer())

            regression = tf.matmul(tanh_output, a) + b

            tf.add_to_collection(name=self.parameters, value=a)
            tf.add_to_collection(name=self.parameters, value=b)

        return regression

    def conv2d(self, x, W, b, strides=1):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def conv_net(self, input):
        F = [16, 16, 3, 6]
        strides = [1, 2, 2, 1]
        padding = 'VALID'

        x = tf.reshape(input, shape=[-1, 85, 72, 1])

        with tf.variable_scope('conv1') as conv1_scope:
            filter = tf.get_variable('filter_1', shape=(5, 6, 1, 6),
                                     initializer=tf.contrib.layers.xavier_initializer(seed=0))

            # if padding is "SAME", output shape is [-1, 85, 72, 6]

            bias = tf.get_variable('bias_1', shape=(6,),
                                   initializer=tf.truncated_normal_initializer(stddev=0.05))

            conv_output = self.conv2d(x, filter, bias)

            pooling_output = tf.nn.max_pool(conv_output, ksize=[1, 5, 6, 1], strides=[1, 1, 1, 1],
                                            padding='VALID')

            batch_normalized = tf.nn.batch_normalization(pooling_output, mean=0, variance=1, offset=0, scale=100,
                                                         variance_epsilon=1e-4)

            tf.add_to_collection(name=self.parameters, value=filter)
            tf.add_to_collection(name=self.parameters, value=bias)

            # output size is 81 * 67 * 6

            print(batch_normalized.get_shape().as_list())

        with tf.variable_scope('conv2') as conv2_scope:
            filter = tf.get_variable(name='filter_2', shape=(10, 10, 6, 32),
                                     initializer=tf.contrib.layers.xavier_initializer(seed=0))
            bias = tf.get_variable(name='bias_2', shape=(32,),
                                   initializer=tf.constant_initializer(0.0))

            conv_output = self.conv2d(batch_normalized, filter, bias)

            print(conv_output.get_shape().as_list())
            pooling_output = tf.nn.max_pool(conv_output, ksize=[1, 54, 40, 1], strides=[1, 1, 1, 1],
                                            padding='VALID')

            # output is 28, 28, 32

            batch_normalized = tf.nn.batch_normalization(pooling_output, mean=0, variance=1, offset=0, scale=100,
                                                         variance_epsilon=1e-4)

            tf.add_to_collection(name=self.parameters, value=filter)
            tf.add_to_collection(name=self.parameters, value=bias)

            print(batch_normalized.get_shape().as_list())

        with tf.variable_scope('fc1') as fc_1:
            fc_dimension = 28 * 28 * 32
            fc1 = tf.reshape(batch_normalized, shape=[-1, fc_dimension])

            weight_fc = tf.get_variable('fc_w_1', shape=(fc_dimension, 1),
                                        initializer=tf.contrib.layers.xavier_initializer(seed=0))

            bias_fc = tf.get_variable('fc_b_1', shape=(),
                                      initializer=tf.constant_initializer(0.0))

            out = tf.matmul(fc1, weight_fc) + bias_fc

            tf.add_to_collection(name=self.parameters, value=weight_fc)
            tf.add_to_collection(name=self.parameters, value=bias_fc)

        return out

    def split_test_train(self):
        '''
        Split train dataset to train, validation, test 
        :return: [train_indices], [validation_indices], [test_indices]
        '''

        shuffled_indices = np.random.permutation(range(1, 10000))

        for i in range(100):
            shuffled_indices = np.random.permutation(shuffled_indices)

        shuffled_indices = np.random.choice(np.random.permutation(range(1, 10000)), self.config.train_data_size)

        test_ratio = 0.01
        train_ratio = (1 - test_ratio) * .8
        validation_ratio = (1 - test_ratio) * .2

        train_num = int(self.config.train_data_size * train_ratio)
        validation_num = int(self.config.train_data_size * validation_ratio)
        test_num = int(self.config.train_data_size * test_ratio)

        train_indices = shuffled_indices[0: train_num]
        validation_indices = shuffled_indices[train_num: train_num + validation_num]
        test_indices = shuffled_indices[-1 * test_num: ]

        return train_indices, validation_indices, test_indices

    @draw_performance.track_plot
    def train_one_epoch(self, sess):
        '''
        Train one epoch, run 10000/batch_size time, each time use the train data with bath_size.
        :return: loss value
        '''


        losses = []
        RMSEs = []
        val_RMSEs = []

        start = 0
        self.config.batch_size = int(len(self.train_indices) * 0.2)

        total_batches = int((len(self.train_indices) - start) / self.config.batch_size)

        for b in range(total_batches):
            indices = np.random.choice(self.train_indices,
                                       self.config.batch_size, replace=True)
            # indices = self.train_indices[start: start+self.config.batch_size]
            start += self.config.batch_size

            train_data, train_labels = self.get_data_by_indices(indices)

            if train_data is None:
                print('training data error, skip this batch')
                continue

            feed_dict = {self.X_train: train_data, self.labels: train_labels}

            # bias = self.bias_1.eval()
            # print('bias before training is: {}'.format(bias))

            L, _, labels, yhat = sess.run(
                [self.loss, self.op, self.labels, self.yhat],
                feed_dict=feed_dict
            )

            # bias = self.bias_1.eval()

            # print('bias: {}'.format(bias))

            losses.append(L)

            y = train_labels
            yhat = self.predict(train_data, sess)

            RMSE_score = self.accuracy(y=y, yhat=yhat)
            RMSEs.append(RMSE_score)

            val_accuracy = self.validation_accuracy(sess)

            val_RMSEs.append(val_accuracy)

            if b % 10 == 0:
                print('')
                print('batching {}/{}'.format(b, total_batches))
                print('train loss: {}'.format(L))
                print('train RMSE score: {}'.format(RMSE_score))
                print('validation RMSE score: {}'.format(val_accuracy))
                print('')
            else:
                print('.', end='')

        # return sum(losses)/len(losses), sum(RMSEs)/len(RMSEs), sum(val_RMSEs)/len(val_RMSEs)
        return losses, RMSEs, val_RMSEs

    def validation_accuracy(self, sess):
        validation_data, validation_labels = self.get_data_by_indices(self.validation_indices)
        y = validation_labels
        yhat = self.predict(validation_data, sess)
        return self.accuracy(y=y, yhat=yhat)

    def get_data_by_indices(self, indices):

        assert (len(indices) == self.config.batch_size) or (len(indices) == len(self.validation_indices))

        train_data = np.zeros(shape=(len(indices), self.X_dimension))
        train_labels = np.zeros(shape=(len(indices),))

        for i, index in enumerate(indices):
            train_data[i,:] = self.cache[index][1]
            train_labels[i] = self.cache[index][0]
        return train_data, train_labels

    # @staticmethod
    # def mean_radar_maps(radar_maps):
    #     mean = np.mean(radar_maps, axis=1)
    #     compressed = mean.flatten()
    #
    #     return compressed

    def compress_radar_maps(self, radar_maps):

        compressed = np.random.choice(radar_maps.flatten(), self.X_dimension)
        return compressed

        # input data is TIME * HEIGHT * 101 * 101, if we get the random submatrix
        # for example 0.2

        # random_matrix = np.random.rand(*radar_maps.shape)
        # random_matrix[random_matrix > self.config.matrix_keep_prob] = 0
        #
        # keep_matrix = radar_maps[random_matrix.nonzero()]

        # time_mean = np.mean(radar_maps, axis=0)
        # height_mean = np.mean(radar_maps, axis=1) # get mean of different height.

        # time_weight = [np.exp(1)] * 5 + [np.exp(2)] * 5 + [np.exp(3)]*5
        # time_weight = [0] * 14 + [1]
        # height_weight = [0] * 3 + [1]
        # height_weight = np.array(height_weight)
        # height_weight = height_weight / np.sum(height_weight)
        #
        # for t in range(time_mean.shape[0]):
        #     time_mean[t] *= height_weight[t]

        #
        # height_weighted_mean = np.mean(time_mean, axis=0)
        # time_height_mean = np.mean(radar_maps, axis=(0, 1))
        # time_mean = np.mean(radar_maps, axis=0)
        # _4th_ = time_mean[3]
        # time_height_mean = _4th_
        # x, y = time_height_mean.shape
        # start_x = x // 2 - (self.config.crop_center // 2)
        # start_y = y // 2 - (self.config.crop_center // 2)
        #
        # start_x = 0
        # start_y = 0
        # cropped = time_height_mean[start_y: start_y+self.config.crop_center, start_x: start_x+self.config.crop_center]
        # compressed = cropped.flatten()

        # compressed = time_height_mean.flatten()

        # compressed = radar_maps.flatten()

    def train(self):
        avg_losses = []
        avg_RMSEs = []
        avg_val_RMSEs = []

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            for ep in range(self.config.epoch):
                start = time.time()
                print('*'*8)
                print(np.random.choice(EMOJIS)+'  epoch: {}'.format(ep))
                # avg_loss, avg_rmse, avg_val_rmse = self.train_one_epoch(sess)
                _loss, _rmse, _val_rmse = self.train_one_epoch(sess)
                print(np.random.choice(EMOJIS)+'  epoch loss: {}'.format(np.mean(_loss)))
                print(np.random.choice(EMOJIS)+'  epoch RMSE: {}'.format(np.mean(_rmse)))
                print(np.random.choice(EMOJIS)+'  epoch validation RMSE: {}'.format(np.mean(_val_rmse)))
                # avg_losses.append(avg_loss)
                # avg_RMSEs.append(avg_rmse)
                # avg_val_RMSEs.append(avg_val_rmse)

                avg_losses += _loss
                avg_RMSEs += _rmse
                avg_val_RMSEs += _rmse

                end = time.time()

                print(np.random.choice(EMOJIS)+'  Total Time: {}'.format(end - start))

        print('done!')

        return avg_losses, avg_RMSEs, avg_val_RMSEs

    def predict(self, test_X, sess):
        yhat = sess.run([self.yhat], feed_dict={self.X_train: test_X})
        return yhat

    def loss(self, yhat):
        epsilon = 1e-3

        # loss_mse = tf.reduce_mean(tf.abs(self.labels - yhat) ** 2)
        loss_mse = tf.sqrt(tf.reduce_mean(tf.square(self.labels - yhat)))
        l2_loss = tf.get_collection('l2_loss')[0] # ~ 0
        L = loss_mse + self.config.regularization_rate * (l2_loss)

        return L

    def optimizer(self, L):
        global_step = tf.Variable(0, trainable=True)
        starter_learning_rate = self.config.learning_rate
        # learning_rate = starter_learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   5000, 0.96, staircase=True)

        op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=L)

        return op

    def accuracy(self, y, yhat):
        return evalution.RMSE(y, yhat)


def dtest():
    rain_regression = RainRegression()

    print('start testing')

    assert rain_regression is not None

    test_radar_maps = np.zeros(shape=(15, 4, 101, 101))

    random_numbers = []
    # for i, j in itertools.product(range(15), range(4)):
    #     random_num = np.random.randint(100)
    #     test_radar_maps[i][j][0][0] = random_num
    #     random_numbers.append(random_num)
    #
    # compressed_radar = rain_regression.mean_radar_maps(test_radar_maps)
    #
    # assert compressed_radar[0] == np.mean(np.array(random_numbers))

    # loss, RMSE, valRMSE = rain_regression.train_one_epoch()
    #
    # assert isinstance(loss, float)


    epoch_losses = rain_regression.train()

    assert epoch_losses is not None
    assert isinstance(epoch_losses[0][0], float)

    del rain_regression

    print('test done!')

# dtest()


def save_train_trace(loss, RMSE, val_RMSE):
    trace = {'loss': loss, 'RMSE': RMSE, 'val_RMSE': val_RMSE}

    with open('trace.pickle', 'wb') as f:
        pickle.dump(trace, f)

    print('trace loaded!')


if __name__ == '__main__':
    print('begin training..')

    with tf.Graph().as_default() as graph:

        rain_regression = RainRegression(test=False)
        losses, RMSEs, val_RMSE = rain_regression.train()

    save_train_trace(losses, RMSEs, val_RMSE)
