import tensorflow as tf
import tensorflow.contrib as tc


class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.lstm_cell = tc.rnn.LayerNormBasicLSTMCell(64)
        else:
            self.lstm_cell = tc.rnn.BasicLSTMCell(64)

    def __call__(self, obss, reuse=False):
        '''
        :param obss: N*t*Obs_dim
        :param reuse: if reuse old variables
        :return: [N*1]*t
        '''
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obss

            x = tf.unstack(x, x.shape(1), axis=1)

            x, state = tf.nn.dynamic_rnn(self.lstm_cell, x, dtype=tf.float32)

            outputs, x = tf.nn.dynamic_rnn(self.lstm_cell, x, dtype=tf.float32)
            shape = tf.shape(outputs)
            shape[2] = self.nb_actions
            outputs = tf.reshape(
                tf.layers.dense(tf.reshape(outputs, [-1, 64]),
                                self.nb_actions,
                                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)),
                shape)

            outputs = tf.nn.tanh(outputs)
        return outputs


class Critic(Model):
    def __init__(self, name='critic', layer_norm=True):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.lstm_cell = tc.rnn.LayerNormBasicLSTMCell(64)
        else:
            self.lstm_cell = tc.rnn.BasicLSTMCell(64)


    def __call__(self, obss, actions, reuse=False):
        '''

        :param obss: N*t*Obs_dim
        :param actions: N*t*Act_dim
        :param reuse: if reuse old variables
        :return: [N*1]*t
        '''

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obss

            x = tf.concat([x, actions], axis=-1)
            x = tf.unstack(x, x.shape(1), axis=1)

            outputs, x = tf.nn.dynamic_rnn(self.lstm_cell, x, dtype=tf.float32)
            shape = tf.shape(outputs)
            shape[2] = 1
            outputs = tf.reshape(
                tf.layers.dense(tf.reshape(outputs, [-1, 64]),
                                1,
                                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)),
                shape)
        return outputs

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
