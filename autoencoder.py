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

from utils import chunks, split


PAD_ID, GO_ID, UNK_D = _RESERVED = range(3)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('filename')
    arg('--state-size', type=int, default=100)
    arg('--batch-size', type=int, default=64)
    arg('--max-seq-length', type=int, default=60)  # TODO - buckets?
    arg('--n-steps', type=int, default=100000)
    arg('--report-step', type=int, default=100)
    arg('--min-char-count', type=int, default=100)
    arg('--n-layers', type=int, default=1)
    arg('--max-gradient-norm', type=float, default=5.0)
    arg('--reverse', action='store_true', help='reverse input')
    arg('--words', action='store_true', help='encode only single words')
    arg('--load', help='restore model from given file')
    arg('--save', help='save model to given file (plus step number)')
    arg('--predict', action='store_true')
    arg('--evaluate', action='store_true')
    args = parser.parse_args()
    print args
    random.seed(1)
    inputs, char_to_id = _read_inputs(args)
    random.shuffle(inputs)
    train_inputs, valid_inputs = split(inputs, 0.9, max_valid=2000)
    input_size = len(char_to_id) + len(_RESERVED)
    print '{} chars, train set size {}, valid set size {}'.format(
       input_size, len(train_inputs), len(valid_inputs))

    model = Model(input_size, args)
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
                feed_dict = model.prepare_batch(batch)
                outputs = sess.run(model.decoder_outputs, feed_dict)
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
            model.train(sess, saver, train_inputs, valid_inputs)


class Model(object):
    def __init__(self, input_size, args):
        self.input_size = input_size
        self.args = args
        self.batch_size = args.batch_size
        cell = rnn_cell.LSTMCell(
            args.state_size, input_size, num_proj=input_size)
        if args.n_layers > 1:
            cell = rnn_cell.MultiRNNCell([cell] * args.n_layers)
        self.encoder_inputs, self.decoder_inputs = [[
            tf.placeholder(tf.float32, shape=[None, input_size],
                        name='{}{}'.format(name, i))
            for i in xrange(length)] for name, length in [
                ('encoder', self.args.max_seq_length),
                ('decoder', self.args.max_seq_length)]]
        # TODO - maybe also use during training,
        # to avoid building one-hot representation (just an optimization).
        # Another (maybe better) way to do is described here
        # https://www.tensorflow.org/versions/master/tutorials/mnist/tf/index.html#loss
        embeddings = tf.constant(np.eye(input_size), dtype=tf.float32)
        loop_function = None
        if args.predict:
            def loop_function(prev, _):
                prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
                return tf.nn.embedding_lookup(embeddings, prev_symbol)
        self.decoder_outputs, _ = seq2seq.tied_rnn_seq2seq(
            self.encoder_inputs, self.decoder_inputs, cell,
            loop_function=loop_function)
        # TODO - add weights
        targets = self.decoder_inputs[1:]
        # FIXME - this scaling by max_seq_length does not take
        # padding into account (see also weights)
        self.decoder_loss = (1. / self.args.max_seq_length) * \
            tf.reduce_mean(tf.add_n([
                tf.nn.softmax_cross_entropy_with_logits(
                    logits, target, name='seq_loss_{}'.format(i))
                for i, (logits, target) in enumerate(
                    zip(self.decoder_outputs, targets))]))
        tf.scalar_summary('train loss', self.decoder_loss)
        self.valid_loss = 1.0 * self.decoder_loss  # FIXME
        tf.scalar_summary('valid loss', self.valid_loss)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer()
        params = tf.trainable_variables()
        gradients = tf.gradients(self.decoder_loss, params)
        clipped_gradients, _norm = tf.clip_by_global_norm(
            gradients, self.args.max_gradient_norm)
        # TODO - monitor norm
        self.train_op = optimizer.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step)
        self.summary_op = tf.merge_all_summaries()

    def prepare_batch(self, inputs):
        ''' Prepare batch for training: return batch_inputs and batch_outputs,
        where each is a list of float32 arrays of shape (batch_size, input_size),
        adding padding and "GO" symbol.
        '''
        batch_size = len(inputs)
        batch_inputs, batch_outputs = [
            [np.zeros([batch_size, self.input_size], dtype=np.float32)
            for _ in xrange(self.args.max_seq_length)] for _ in xrange(2)]
        for n_batch, input_ in enumerate(inputs):
            n_pad = (self.args.max_seq_length - len(input_))
            padded_input = [PAD_ID] * n_pad + list(
                reversed(input_) if self.args.reverse else input_)
            for values, seq in [
                    (batch_inputs, [padded_input]),
                    (batch_outputs, [[GO_ID], input_ , repeat(PAD_ID, n_pad - 1)])
                    ]:
                for i, id_ in enumerate(chain(*seq)):
                    values[i][n_batch][id_] = 1.0
        feed_dict = {
            var.name: val for var, val in
            chain(izip(self.encoder_inputs, batch_inputs),
                  izip(self.decoder_inputs, batch_outputs))}
        return feed_dict

    def train(self, sess, saver, train_inputs, valid_inputs):
        losses = []
        summary_writer = None
        if self.args.save:
            summary_writer = tf.train.SummaryWriter(
                self.args.save, flush_secs=10)
        t0 = time.time()
        for i in xrange(self.args.n_steps):
            loss = self._train_step(sess, train_inputs, summary_writer)
            losses.append(loss)
            step = self.global_step.eval()
            if i == 0 or step % self.args.report_step == 0:
                print '{:>5}: train loss {:.4f}, valid loss {:.4f} in {} s'\
                    .format(
                        step,
                        np.mean(losses),
                        self._valid_loss(sess, valid_inputs, summary_writer),
                        int(time.time() - t0))
                losses = []
                if self.args.save:
                    saver.save(sess, self.args.save, global_step=step)
                if self.args.evaluate:
                    break

    def _train_step(self, sess, inputs, summary_writer):
        b_inputs = [random.choice(inputs) for _ in xrange(self.args.batch_size)]
        feed_dict = self.prepare_batch(b_inputs)
        ops = [self.decoder_loss, self.summary_op]
        if not self.args.evaluate:
            ops.append(self.train_op)
        loss, summary_str = sess.run(ops, feed_dict)[:2]
        step = self.global_step.eval()
        if summary_writer and step % 10 == 0:
            summary_writer.add_summary(summary_str, step)
        return loss

    def _valid_loss(self, sess, valid_inputs, summary_writer):
        loss, summary_str = sess.run(
            [self.valid_loss, self.summary_op],
            feed_dict=self.prepare_batch(valid_inputs))
        if summary_writer:
            summary_writer.add_summary(summary_str, self.global_step.eval())
        return loss


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


if __name__ == '__main__':
    main()
