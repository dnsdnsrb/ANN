import tensorflow as tf
import numpy as np
import NNutils
import mnist
from datetime import datetime
import flags
class Network():
    def __init__(self, layers = [400, 200, 100], regul=False, bn=False, droprate=.0):
        self.batch_size = 128
        self.learning_rate = 0.001
        self.global_step = tf.Variable(0, trainable=False)

        self.x = tf.placeholder("float32", [None, 28 * 28 * 1])
        self.y = tf.placeholder("float32", [None, 10])

        self.droprate_train = droprate
        self.droprate_test = 0.0
        self.droprate_ph = tf.placeholder("float")

        self.layers = layers
        self.train_list = []

        self.activation = "relu"
        self.network = "mlp"

        #정규화 관련 변수
        if regul == True:
            self.regularize = 0.001
            self.regularizer = tf.contrib.layers.l2_regularizer(self.regularize)
            self.network += "_regul"
        else:
            self.regularizer = None

        if bn == True:
            self.batch_normalization = True
            self.network += "_bn"
        else:
            self.batch_normalization = False

    def fc_layer(self, x, output_num, activation, dropout):
        if activation == 'relu':
            activation = tf.nn.relu
        elif activation == 'sigmoid':
            activation = tf.nn.sigmoid
        elif activation == 'tanh':
            activation = tf.nn.tanh
        else:
            activation == None

        output = tf.layers.dense(x, output_num, activation=activation,
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                 kernel_regularizer=self.regularizer)
        output = tf.nn.dropout(output, 1 - dropout)

        y = output
        return y

    def info(self):
        layers_sum = str(sum(self.layers))
        layers_num = str(len(self.layers))
        dropout = str(self.droprate_train)
        return self.network + "-" + self.activation + "-" + layers_num + "-" + layers_sum + "-d" + dropout

    def model(self, x): #[3072, 450, 300, 200, 100, 50, 10]
        layers = self.layers
        image_size = 32
        output = x
        reshape_size = 0

        layer_num = 0

        for layer in layers:
            layer_num += 1
            with tf.variable_scope('fc' + str(layer_num)):
                if self.batch_normalization == True:
                    output = tf.layers.batch_normalization(output)
                # output = fc_layer('fc', output, layer, activation=self.activation, dropout=self.dropout_normal)
                output = self.fc_layer(output, layer, activation=self.activation, dropout=self.droprate_ph)
        with tf.variable_scope('fc'):
            output = tf.contrib.layers.fully_connected(output, 10, activation_fn=None)

        y = output
        return y

    def train(self):

        #learning rate
        with tf.name_scope("learning_rate"):
            learning_rate = tf.train.exponential_decay(0.001,
                                                       self.global_step,
                                                       (50000 / self.batch_size) * 10,
                                                       0.95, staircase=True)
            learning_rate = tf.maximum(0.0001, learning_rate)
            tf.summary.scalar("learning_rate", learning_rate)

        #model
        y_ = self.model(self.x)

        #cost and training
        with tf.name_scope("cost"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=self.y))

            #정규항 추가
            if self.regularizer != None:
                reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                reg_term = tf.contrib.layers.apply_regularization(self.regularizer, reg_variables)
                self.loss += reg_term

            self.training = tf.train.AdamOptimizer(learning_rate=learning_rate). \
                minimize(self.loss, global_step=self.global_step)

            tf.summary.scalar("loss", self.loss)

        with tf.name_scope("accuracy"):
            compare = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(compare, "float"))

            tf.summary.scalar("accuarcy", self.accuracy)

    def run(self, step_limit):
        self.train()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            dataset = mnist.read_data_sets(flags.MNIST_DIR, one_hot=True)
            train_data, train_label, test_data, test_label = dataset.train.images, dataset.train.labels, \
                                          dataset.test.images, dataset.test.labels

            test_indices = np.arange(len(test_data))
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:1000]
            name = self.info()
            path = "mnist/" + str(step_limit) + name
            saver = NNutils.save(path, sess)
            writer, writer_test, merged = NNutils.graph(path, sess)

            step = sess.run(self.global_step)
            while step < step_limit:
                print("step :", step)
                for start, end in zip(range(0, len(train_data), self.batch_size),
                                      range(self.batch_size, len(train_data), self.batch_size)):
                    summary, \
                    _, loss, \
                    step = sess.run([merged,
                                     self.training, self.loss,
                                     self.global_step],
                                    feed_dict={self.x: train_data[start:end],
                                               self.y: train_label[start:end],
                                               self.droprate_ph: self.droprate_train})

                    if step % 50 == 0:
                        writer.add_summary(summary, step)
                        print(step, datetime.now(), loss)

                summary, \
                loss, \
                accuracy = sess.run([merged, self.loss, self.accuracy],
                                    feed_dict={self.x: test_data,
                                               self.y: test_label,
                                               self.droprate_ph: self.droprate_test})

                writer_test.add_summary(summary, step)
                print("test results : ", accuracy, loss)
                saver.save(sess, path + "/" + name + ".ckpt", step)

if __name__ == "__main__":
    model = Network(regul=False, bn=False, droprate=0.5)
    model.run(100000)