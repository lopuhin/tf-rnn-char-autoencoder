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
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('filename')
    arg('--state_size', type=int, default=100)
    arg('--batch_size', type=int, default=32)
    arg('--max_seq_length', type=int, default=20)  # TODO - buckets?
    arg('--n-steps', type=int, default=10000)
    args = parser.parse_args()
    print 'reading inputs'
    inputs, char_to_id = _read_inputs(args.filename, args.max_seq_length)
    input_size = len(char_to_id)
    print 'input_size', input_size

    encoder_inputs, decoder_inputs, encoder_outputs, encoder_states = \
        _create_model(input_size, args)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for step in xrange(args.n_steps):
            feed_dict = {}
            batch_inputs, batch_outputs = _prepare_batch(
                inputs, input_size, args)
            feed_dict = {
                var.name: val for var, val in
                chain(izip(encoder_inputs, batch_inputs),
                      izip(decoder_inputs, batch_outputs))}
            # TODO - loss?
            outputs = sess.run(encoder_outputs, feed_dict)


def _create_model(input_size, args):
    cell = rnn_cell.LSTMCell(args.state_size, input_size)
    encoder_inputs, decoder_inputs = [[
        tf.placeholder(tf.float32, shape=[args.batch_size, input_size],
                       name='{}{}'.format(name, i))
        for i in xrange(length)] for name, length in [
            ('encoder', args.max_seq_length),
            ('decoder', args.max_seq_length)]]
    outputs, states = seq2seq.tied_rnn_seq2seq(
        encoder_inputs, decoder_inputs, cell)
    return encoder_inputs, decoder_inputs, outputs, states


def _read_inputs(filename, max_seq_length):
    ''' Return a list of inputs (int lists), and an encoding dict.
    '''
    char_to_id = {}
    word_re = re.compile(r'\w+', re.U)
    inputs = []
    with codecs.open(filename, 'rb', 'utf-8') as f:
        for line in f:
            for ch in line:
                if ch not in char_to_id:
                    char_to_id[ch] = len(char_to_id) + len(_RESERVED)
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


def _prepare_batch(inputs, input_size, args):
    ''' Prepare batch for training: return batch_inputs and batch_outputs,
    where each is a list of float32 arrays of shape (batch_size, input_size),
    adding padding and "GO" symbol.
    '''
    batch_inputs, batch_outputs = [
        [np.zeros([args.batch_size, input_size], dtype=np.float32)
         for _ in xrange(args.max_seq_length)] for _ in xrange(2)]
    for n_batch in xrange(args.batch_size):
        input_ = random.choice(inputs)
        n_pad = (args.max_seq_length - len(input_))
        for values, seq in [
                # TODO - reverse inputs?
                (batch_inputs, [input_, repeat(PAD_ID, n_pad)]),
                (batch_outputs, [[GO_ID], input_ , repeat(PAD_ID, n_pad - 1)])
                ]:
            for i, id_ in enumerate(chain(*seq)):
                values[i][n_batch] = id_
    return batch_inputs, batch_outputs


if __name__ == '__main__':
    main()
