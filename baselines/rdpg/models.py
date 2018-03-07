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
        self.lstm_cell = tc.rnn.BasicLSTMCell(64, forget_bias=1.0)

        self.decoding_weights = tf.Variable(tf.random_normal([64, nb_actions]))
        self.decoding_biases = tf.Variable(tf.random_normal([nb_actions]))

    def __call__(self, obss, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obss

            x = tf.unstack(x, x.shape(1), axis=1)

            x, state = tc.rnn.static_rnn(self.lstm_cell, x, dtype=tf.float32)

            for i in range(len(x)):
                x[i] = tf.matmul(x[i], self.decoding_weights) + self.decoding_biases
                x[i] = tf.nn.tanh(x[i])

        return x


class Critic(Model):
    def __init__(self, name='critic', layer_norm=True):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.lstm_cell = tc.rnn.BasicLSTMCell(64, forget_bias=1.0)

        self.decoding_weights = tf.Variable(tf.random_normal([64, 1]))
        self.decoding_biases = tf.Variable(tf.random_normal([1]))

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

            x, state = tc.rnn.static_rnn(self.lstm_cell, x, dtype=tf.float32)

            for i in range(len(x)):
                x[i] = tf.matmul(x[i], self.decoding_weights) + self.decoding_biases

        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
