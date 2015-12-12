#!/usr/bin/env python
# encoding: utf-8

import re
import codecs
import argparse
import random
from itertools import chain, repeat, izip

import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import seq2seq, rnn_cell


PAD_ID, GO_ID = _RESERVED = 0, 1


def main():
    random.seed(1)
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('filename')
    arg('--state_size', type=int, default=100)
    arg('--batch_size', type=int, default=32)
    arg('--max_seq_length', type=int, default=20)  # TODO - buckets?
    arg('--n-steps', type=int, default=10000)
    arg('--report-step', type=int, default=100)
    arg('--reverse', action='store_true', help='reverse input')
    args = parser.parse_args()
    print 'reading inputs'
    inputs, char_to_id = _read_inputs(args.filename, args.max_seq_length)
    input_size = len(char_to_id)
    print 'input_size', input_size

    encoder_inputs, decoder_inputs, decoder_outputs, decoder_loss = \
        _create_model(input_size, args)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(decoder_loss)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        losses = []
        for step in xrange(args.n_steps):
            feed_dict = {}
            b_inputs = [random.choice(inputs) for _ in xrange(args.batch_size)]
            batch_inputs, batch_outputs = _prepare_batch(
                b_inputs, input_size, args.max_seq_length,
                reverse=args.reverse)
            feed_dict = {
                var.name: val for var, val in
                chain(izip(encoder_inputs, batch_inputs),
                      izip(decoder_inputs, batch_outputs))}
            _, loss = sess.run([train_op, decoder_loss], feed_dict)
            losses.append(loss)
            if step % args.report_step == 1:
                print '{}: {}'.format(
                    int(step / args.report_step), np.mean(losses))
                losses = []


def _create_model(input_size, args):
    cell = rnn_cell.LSTMCell(args.state_size, input_size, num_proj=input_size)
    encoder_inputs, decoder_inputs = [[
        tf.placeholder(tf.float32, shape=[args.batch_size, input_size],
                       name='{}{}'.format(name, i))
        for i in xrange(length)] for name, length in [
            ('encoder', args.max_seq_length),
            ('decoder', args.max_seq_length)]]
    decoder_outputs, _ = seq2seq.tied_rnn_seq2seq(
        encoder_inputs, decoder_inputs, cell)
    # TODO - add weights
    targets = decoder_inputs[1:]
    decoder_loss = tf.reduce_mean(tf.add_n([
        tf.nn.softmax_cross_entropy_with_logits(
            logits, target, name='seq_loss_{}'.format(i))
        for i, (logits, target) in enumerate(zip(decoder_outputs, targets))]))
    return encoder_inputs, decoder_inputs, decoder_outputs, decoder_loss


def _read_inputs(filename, max_seq_length):
    ''' Return a list of inputs (int lists), and an encoding dict.
    '''
    char_to_id = {}
    word_re = re.compile(r'\w+', re.U)
    inputs = []
    with codecs.open(filename, 'rb', 'utf-8') as f:
        for line in f:
            for word in word_re.findall(line):
                if len(word) < max_seq_length:  # one more for "GO"
                    inputs.append(_encode(word, char_to_id))
    return inputs, char_to_id


def _encode(string, char_to_id):
    result = []
    for ch in string:
        id_ = char_to_id.get(ch)
        if id_ is None:
            char_to_id[ch] = len(char_to_id) + len(_RESERVED)
        result.append(id_)
    return result


def _prepare_batch(inputs, input_size, max_seq_length, reverse=False):
    ''' Prepare batch for training: return batch_inputs and batch_outputs,
    where each is a list of float32 arrays of shape (batch_size, input_size),
    adding padding and "GO" symbol.
    '''
    batch_size = len(inputs)
    batch_inputs, batch_outputs = [
        [np.zeros([batch_size, input_size], dtype=np.float32)
         for _ in xrange(max_seq_length)] for _ in xrange(2)]
    for n_batch, input_ in enumerate(inputs):
        n_pad = (max_seq_length - len(input_))
        padded_input = list(input_) + [PAD_ID] * n_pad
        if reverse:
            padded_input.reverse()
        for values, seq in [
                (batch_inputs, [padded_input]),
                (batch_outputs, [[GO_ID], input_ , repeat(PAD_ID, n_pad - 1)])
                ]:
            for i, id_ in enumerate(chain(*seq)):
                values[i][n_batch][id_] = 1.0
    return batch_inputs, batch_outputs


if __name__ == '__main__':
    main()
