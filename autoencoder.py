#!/usr/bin/env python
# encoding: utf-8

import re
import codecs
import argparse
import random
import time
from itertools import chain, repeat, izip
from collections import Counter
import cPickle as pickle

import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import seq2seq, rnn_cell

from utils import chunks


PAD_ID, GO_ID, UNK_D = _RESERVED = range(3)


def main():
    random.seed(1)
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('filename')
    arg('--state-size', type=int, default=100)
    arg('--batch-size', type=int, default=64)
    arg('--max-seq-length', type=int, default=100)  # TODO - buckets?
    arg('--n-steps', type=int, default=100000)
    arg('--report-step', type=int, default=100)
    arg('--min-char-count', type=int, default=100)
    arg('--n-layers', type=int, default=1)
    arg('--reverse', action='store_true', help='reverse input')
    arg('--words', action='store_true', help='encode only single words')
    arg('--load', help='restore model from given file')
    arg('--save', help='save model to given file (plus step number)')
    arg('--predict', action='store_true')
    arg('--evaluate', action='store_true')
    args = parser.parse_args()
    print args
    inputs, char_to_id = _read_inputs(args)
    input_size = len(char_to_id) + len(_RESERVED)
    print 'input_size', input_size

    encoder_inputs, decoder_inputs, decoder_outputs, decoder_loss = \
        _create_model(input_size, args)
    optimizer = tf.train.AdamOptimizer()
    # TODO - monitor gradient norms, clip them?
    train_op = optimizer.minimize(decoder_loss)
    saver = tf.train.Saver(tf.all_variables())

    with tf.Session() as sess:
        if args.load:
            saver.restore(sess, args.load)
        else:
            sess.run(tf.initialize_all_variables())
        if args.predict:
            id_to_char = {id_: ch for ch, id_ in char_to_id.iteritems()}
            for id_ in _RESERVED:
                id_to_char[id_] = ''
            for batch in chunks(inputs, args.batch_size):
                feed_dict = _prepare_batch(
                    batch, input_size, args.max_seq_length,
                    encoder_inputs, decoder_inputs, reverse=args.reverse)
                outputs = sess.run(decoder_outputs, feed_dict)
                input_lines = [[id_to_char[id_] for id_ in line]
                               for line in batch]
                output_lines = [[] for _ in xrange(args.batch_size)]
                for char_block in outputs:
                    for i, id_ in enumerate(np.argmax(char_block, axis=1)):
                        output_lines[i].append(id_to_char[id_])
                for inp, out in zip(input_lines, output_lines):
                    print
                    print ''.join(inp)
                    print ''.join(out)
                import pdb; pdb.set_trace()
        else:
            _train(inputs, input_size, args, sess, saver,
                encoder_inputs, decoder_inputs, train_op, decoder_loss)


def _train(inputs, input_size, args, sess, saver,
        encoder_inputs, decoder_inputs, train_op, decoder_loss):
    def _step(step):
        feed_dict = {}
        b_inputs = [random.choice(inputs) for _ in xrange(args.batch_size)]
        feed_dict = _prepare_batch(
            b_inputs, input_size, args.max_seq_length,
            encoder_inputs, decoder_inputs, reverse=args.reverse)
        ops = [decoder_loss]
        if not args.evaluate:
            ops.append(train_op)
        loss = sess.run(ops, feed_dict)[0]
        losses.append(loss)
        if step % args.report_step == 1:
            print '{:>3}: loss {:.4f} in {} s'.format(
                int(step / args.report_step),
                np.mean(losses),
                int(time.time() - t0))
            losses[:] = []
            if args.save:
                saver.save(sess, args.save, global_step=step)
            if args.evaluate:
                return False
        return True
    losses = []
    t0 = time.time()
    for n in xrange(args.n_steps):
        if not _step(n):
            break


def _create_model(input_size, args):
    cell = rnn_cell.LSTMCell(args.state_size, input_size, num_proj=input_size)
    if args.n_layers > 1:
        cell = rnn_cell.MultiRNNCell([cell] * args.n_layers)
    encoder_inputs, decoder_inputs = [[
        tf.placeholder(tf.float32, shape=[None, input_size],
                       name='{}{}'.format(name, i))
        for i in xrange(length)] for name, length in [
            ('encoder', args.max_seq_length),
            ('decoder', args.max_seq_length)]]
    # TODO - maybe also use during training,
    # to avoid building one-hot representation (just an optimization).
    embeddings = tf.constant(np.eye(input_size), dtype=tf.float32)
    loop_function = None
    if args.predict:
        def loop_function(prev, _):
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embeddings, prev_symbol)
    decoder_outputs, _ = seq2seq.tied_rnn_seq2seq(
        encoder_inputs, decoder_inputs, cell, loop_function=loop_function)
    # TODO - add weights
    targets = decoder_inputs[1:]
    # TODO - this scaling by max_seq_length does not take padding into account
    decoder_loss = (1. / args.max_seq_length) * tf.reduce_mean(tf.add_n([
        tf.nn.softmax_cross_entropy_with_logits(
            logits, target, name='seq_loss_{}'.format(i))
        for i, (logits, target) in enumerate(zip(decoder_outputs, targets))]))
    return encoder_inputs, decoder_inputs, decoder_outputs, decoder_loss


def _read_inputs(args):
    ''' Return a list of inputs (int lists), and an encoding dict.
    '''
    word_re = re.compile(r'\w+', re.U)
    with codecs.open(args.filename, 'rb', 'utf-8') as textfile:
        if args.load:
            with open(args.load.rsplit('-', 1)[0] + '.mapping', 'rb') as f:
                char_to_id = pickle.load(f)
        else:
            char_counts = Counter(ch for line in textfile for ch in line)
            char_to_id = {
                ch: id_ for id_, ch in enumerate(
                    (ch for ch, count in char_counts.iteritems()
                        if count >= args.min_char_count),
                    len(_RESERVED))}
            textfile.seek(0)
        if args.save:
            with open(args.save + '.mapping', 'wb') as f:
                pickle.dump(char_to_id, f, protocol=-1)
        def inputs_iter():
            for line in textfile:
                line = line.strip()
                if args.words:
                    for word in word_re.findall(line):
                        yield word + ' '
                else:
                    yield line
        inputs = []
        for string in inputs_iter():
            limit = args.max_seq_length - 1  # one more for "GO"
            if len(string) > limit:
                string = string[:limit - 1].rsplit(None, 1)[0] + ' '
            if len(string) <= limit:
                inputs.append([char_to_id.get(ch, UNK_D) for ch in string])
    return inputs, char_to_id


def _prepare_batch(inputs, input_size, max_seq_length,
        encoder_inputs, decoder_inputs, reverse=False):
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
        padded_input = [PAD_ID] * n_pad + list(
            reversed(input_) if reverse else input_)
        for values, seq in [
                (batch_inputs, [padded_input]),
                (batch_outputs, [[GO_ID], input_ , repeat(PAD_ID, n_pad - 1)])
                ]:
            for i, id_ in enumerate(chain(*seq)):
                values[i][n_batch][id_] = 1.0
    feed_dict = {
        var.name: val for var, val in
        chain(izip(encoder_inputs, batch_inputs),
              izip(decoder_inputs, batch_outputs))}
    return feed_dict


if __name__ == '__main__':
    main()
