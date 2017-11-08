from __future__ import division
import sys
import json
import generator
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from sklearn.metrics import confusion_matrix


class Classifier:
    # This classifies if the sentence can be framed into a question or not
    def __init__(self, config_path):
        np.random.seed(0)
        self.config(config_path)
        emb = self.load_word2vec()
        self.embeddings = tf.Variable(tf.constant(emb, dtype='float32', shape=[self.vocabulary_size, self.embedding_size]))



    def config(self, config_path):
        data = json.load(open(config_path))
        print "Config data = ", data
        self.hashed_train_path = data["hashed_train_path"]
        self.hashed_dev_path = data["hashed_dev_path"]
        self.hashed_test_path = data["hashed_test_path"]
        self.embedding_path = data["embed_path"]
        self.vocabulary_path = data["vocab_path"]
        self.embedding_size = data["embedding_size"]
        self.vocabulary_size = data["vocab_size"]
        self.lstm_hidden_size = data["lstm_hidden_size"]
        self.fc_hidden_size = data["fc_hidden_size"]
        self.n_fc = data["n_fc"]
        self.input_size = data["input_size"]
        self.output_size = data["output_size"]
        self.max_sentence_len = data["max_sentence_len"]
        self.activation = data["activation"]
        self.optimizer = data["optimizer"]
        self.learning_rate = data["learning_rate"]
        self.batch_size = data["batch_size"]
        self.n_epochs = data["n_epochs"]
        self.eval_frequency = data["eval_frequency"]
        self.train_size = data["train_size"]
        self.dev_size = data["dev_size"]
        self.test_size = data["test_size"]


    def load_word2vec(self):
        emb = np.random.uniform(low=-1, high=1, size=(self.vocabulary_size, self.embedding_size))
        nlines = 0
        vocabulary_index = json.load(open(self.vocabulary_path))
        with open(self.embedding_path) as f:
            for line in f:
                nlines += 1
                if nlines == 1:
                    continue
                items = line.split()
                term = items[0]
                if vocabulary_index.get(term) == None:
                    continue
                tid = vocabulary_index[term]
                if tid > self.vocabulary_size:
                    print tid
                    continue
                vec = np.array([float(t) for t in items[1:]])
                emb[tid, :] = vec
                if nlines % 20000 == 0:
                    print "load {0} vectors...".format(nlines)
        print "Embedding size = {}".format(emb.shape)
        return emb

    def load_data(self):
        pass

    def model(self, input_data, sequence_lengths):
        # [bs, para_len, sent_len, emb_len]
        input_embed = tf.nn.embedding_lookup(self.embeddings, input_data, name='sent_emb')
        # [bs, para_len, emb_len]
        sum_embed = tf.reduce_sum(input_embed, 2)
        # Unstack sum_embed to get a list of 'timesteps' tensors of shape [batch_size, emb_len]
        # sum_embed_unstack = tf.unstack(sum_embed, self.max_paragraph_len, 1)
        # forward LSTM Cell
        lstm_fw_cell = rnn.BasicLSTMCell(self.lstm_hidden_size, forget_bias=1.0)
        # backward LSTM Cell
        lstm_bw_cell = rnn.BasicLSTMCell(self.lstm_hidden_size, forget_bias=1.0)
        # Get lstm cell output
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                                    lstm_bw_cell, sum_embed,
                                                                    sequence_length=sequence_lengths,
                                                                    dtype=tf.float32)
        context_rep = tf.concat([output_fw, output_bw], axis=-1)
        context_rep_flat = tf.reshape(context_rep, [-1, 2 * self.lstm_hidden_size])

        print "Shapes"
        print "embedding: {}\n embedded input: {}\n sum embedded input: {}\n context_rep: {}".format(
            self.embeddings.get_shape(), input_embed.get_shape(), sum_embed.get_shape(),
            context_rep_flat.get_shape()
        )
        dense_input = context_rep_flat
        for i in range(self.n_fc):
            # Fully connected layer
            if self.activation == "tanh":
                act = tf.nn.tanh
            else:
                act = tf.nn.sigmoid
            dense_output = tf.layers.dense(inputs=dense_input, units=self.fc_hidden_size[i], activation=act)
            dense_input = dense_output


        out = tf.matmul(dense_output, self.weights['out']) + self.biases['out']
        return out


    def train(self):
        # Define weights
        self.weights = {
            # Hidden layer weights => 2*n_hidden because of forward + backward cells
            'out': tf.Variable(tf.random_normal([self.fc_hidden_size[-1], self.output_size]))
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal([self.output_size]))
        }
        input_data = tf.placeholder(tf.int32, shape=[self.batch_size, None, self.max_sentence_len], name="input_data")
        output_data = tf.placeholder(tf.int32, shape=[self.batch_size, None, self.output_size], name="output_data")
        sequence_lengths = tf.placeholder(tf.int32, shape=[self.batch_size])
        logits = self.model(input_data, sequence_lengths)
        predictions = tf.nn.softmax(logits)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=output_data))

        # Evaluate model (with test logits, for dropout to be disabled)
        # correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(tf.reshape(output_data, [-1, self.output_size]), 1))
        # confusion_matrix = tf.confusion_matrix(tf.argmax(tf.reshape(output_data, [-1, self.output_size]), 1), tf.argmax(predictions, 1))

        if self.optimizer == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)

        # Initialize the variables (i.e. assign their default value)
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()

        with tf.Session() as sess:
            # Run the initializer
            sess.run(init_g)
            sess.run(init_l)
            for e in range(self.n_epochs):
                steps = 0
                fp = open(self.hashed_train_path)
                for batch in generator.sentence_classifier_generator(fp, \
                        self.batch_size, self.max_sentence_len, self.output_size):
                    batch_input, batch_output, batch_sequence_lengths, max_paragraph_len = batch
                    train_feed_dict = {input_data: batch_input, output_data: batch_output, sequence_lengths: batch_sequence_lengths}
                    # Run optimization op (backprop)
                    _, l, pred = sess.run([optimizer, loss, predictions], feed_dict=train_feed_dict)
                    cm = self.get_confusion_matrix(pred, batch_output, batch_sequence_lengths, max_paragraph_len)
                    steps += 1
                    if steps % self.eval_frequency == 0:
                        print "Epoch = {} Step = {}".format(e, steps)

                        dev_loss = 0.
                        dev_cm = np.zeros((self.output_size, self.output_size))
                        fp = open(self.hashed_dev_path)
                        dev_size = 0
                        for batch in generator.sentence_classifier_generator(fp, \
                                self.batch_size, self.max_sentence_len, self.output_size):
                            batch_input, batch_output, batch_sequence_lengths, max_paragraph_len = batch
                            dev_feed_dict = {input_data: batch_input, output_data: batch_output, sequence_lengths: batch_sequence_lengths}
                            # Run optimization op (backprop)
                            l, pred = sess.run([loss, predictions], feed_dict=dev_feed_dict)
                            dev_loss += l
                            dev_cm += self.get_confusion_matrix(pred, batch_output, batch_sequence_lengths, max_paragraph_len)
                            dev_size += np.sum(batch_sequence_lengths)
                        precision = dev_cm[1][1]/(dev_cm[1][1] + dev_cm[0][1])
                        recall = dev_cm[1][1]/(dev_cm[1][1] + dev_cm[1][0])
                        accuracy = (dev_cm[0][0]+dev_cm[1][1])/(dev_cm[0][0]+dev_cm[1][1]+dev_cm[0][1]+dev_cm[1][0])
                        print "Dev Batch loss = {} Accuracy = {:.3f} Precision = {:.3f} Recall = {:.3f}".format(dev_loss/dev_size, accuracy, precision, recall)
                        print dev_cm
                        print "dev sent size = {}".format(dev_size)

                        test_loss = 0.
                        test_cm = np.zeros((self.output_size, self.output_size))
                        fp = open(self.hashed_test_path)
                        test_size = 0
                        for batch in generator.sentence_classifier_generator(fp, \
                                self.batch_size, self.max_sentence_len, self.output_size):
                            batch_input, batch_output, batch_sequence_lengths, max_paragraph_len = batch
                            test_feed_dict = {input_data: batch_input, output_data: batch_output, sequence_lengths: batch_sequence_lengths}
                            # Run optimization op (backprop)
                            l, pred = sess.run([loss, predictions], feed_dict=test_feed_dict)
                            test_loss += l
                            test_cm += self.get_confusion_matrix(pred, batch_output, batch_sequence_lengths, max_paragraph_len)
                            test_size += np.sum(batch_sequence_lengths)
                        precision = test_cm[1][1] / (test_cm[1][1] + test_cm[0][1])
                        recall = test_cm[1][1] / (test_cm[1][1] + test_cm[1][0])
                        accuracy = (test_cm[0][0] + test_cm[1][1]) / (test_cm[0][0] + test_cm[1][1] + test_cm[0][1] + test_cm[1][0])
                        print "Test Batch loss = {} Accuracy = {:.3f} Precision = {:.3f} Recall = {:3f}".format(test_loss/test_size, accuracy, precision, recall)
                        print test_cm
                        print "test sent size = {}".format(test_size)

    def get_confusion_matrix(self, predictions, labels, sequence_lengths, max_paragraph_len):
        matrix = np.zeros((self.output_size, self.output_size))
        predictions = predictions.reshape(self.batch_size, max_paragraph_len, self.output_size)
        seq_size = 0
        for b in range(self.batch_size):
            if sequence_lengths[b] == 0:
                continue
            max_pred = predictions[b][:sequence_lengths[b]]
            max_labels = labels[b][:sequence_lengths[b]]
            seq_size += sequence_lengths[b]
            cm = confusion_matrix(np.argmax(max_labels, 1), np.argmax(max_pred, 1), labels=[0,1])
            matrix += cm
            assert sequence_lengths[b] == np.sum(cm)
        assert seq_size == np.sum(matrix)
        return matrix



    def test(self):
        pass


def main(args):
    c = Classifier(args[1])
    c.train()




if __name__ == '__main__':
    main(sys.argv)
