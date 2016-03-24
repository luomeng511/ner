#!/usr/bin/python
# -*- coding: utf8 -*-

# TODO:
# - reformat for input (base on conll standard) : DONE
# - generate batch data :
# - modify session, set up feed_dict
# - calculate accuracy
# ? dimention of matrices
# ! Save and Restore model
# ! Configure run on Spark
# - Prepare test/data

import tensorflow as tf
import numpy as np

filename = 'test.conll'



vocabulary_size = 0

def build_dataset(filename):
    global vocabulary_size
    dataset = list()
    labels = list()
    f = open(filename, 'r')
    sentence = f.read().split('\n')
    vocabulary_size = len(sentence)
    for line in sentence:
        word, label = line.split('_')
        if label == 'O':
            label = np.array([0.0, 0.0, 0.0, 0.1], np.float32)
        elif label == 'PER':
            label = np.array([0.0, 0.0, 0.1, 0.0], np.float32)
        elif label == 'LOCATION':
            label = np.array([0.0, 0.1, 0.0, 0.0], np.float32)
        elif label == 'ORG':
            label = np.array([0.1, 0.0, 0.0, 0.0], np.float32)
        dataset.append(word)
        labels.append(label)
        print word, label
    return dataset, labels

dataset, labels = build_dataset(filename)
# print dataset
# print labels
print vocabulary_size



def generate_batch(batch_size):
    pass



batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
num_nodes = 4  # (PER, ORG, LOCATION, O)
graph = tf.Graph()
with graph.as_default():
    # Parameters:
    # Input gate: input, previous output, and bias.
    ix = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
    im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ib = tf.Variable(tf.zeros([1, num_nodes]))
    # Forget gate: input, previous output, and bias.
    fx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
    fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    fb = tf.Variable(tf.zeros([1, num_nodes]))
    # Memory cell: input, state and bias.
    cx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
    cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    cb = tf.Variable(tf.zeros([1, num_nodes]))
    # Output gate: input, previous output, and bias.
    ox = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
    om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ob = tf.Variable(tf.zeros([1, num_nodes]))
    # Variables saving state across unrollings.
    saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    # Classifier weights and biases.
    w = tf.Variable(tf.truncated_normal([num_nodes, embedding_size], -0.1, 0.1))
    b = tf.Variable(tf.zeros([embedding_size]))


    def lstm_cell(i, o, state):
        """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
        Note that in this formulation, we omit the various connections between the
        previous state and the gates."""
        input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
        forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
        update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
        output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
        state = forget_gate * state + input_gate * tf.tanh(update)
        return output_gate * tf.tanh(state), state


    # Word embedding
        # Make embedding of all words as a table[vocabulary_size, embedding_size] with
        #     vocabulary_size: Number of words in test data
        #     embedding_size : Dimention of embedding vectors
        # train_inputs: contain (window_size) - number of next words in dataset
        # train_labels: labels of inputs, in one-hot form
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    window_size = 5
    train_inputs = list()
    train_labels = list()
    outputs = list()
    output = saved_output
    state = saved_state
    # Does it really need a loop? Can be removed?
    for _ in xrange(window_size):
        train_data = tf.placeholder(tf.int32, shape=[batch_size])
        embed = tf.nn.embedding_lookup(embeddings, train_data)
        label = tf.placeholder(tf.float32, shape=[num_nodes])
        train_inputs.append(embed)
        train_labels.append(label)
        output, state = lstm_cell(embed, output, state)
    logits = tf.nn.xw_plus_b(outputs, w, b)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, train_labels))


   
