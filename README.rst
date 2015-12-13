LSTM Character Autoencoder
==========================

Note that loss is NOT perplexity.

words with reverse (061e6b0ce3ed1129cd437af6b6b2653277a9cd2d)::

    $ ./autoencoder.py ~/programming/ling/corpora/opencorpora.txt \
        --n-steps=10000 --reverse --words --max-seq-length=12
    Namespace(batch_size=32, filename='/Users/kostia/programming/ling/corpora/opencorpora.txt', max_seq_length=12, min_char_count=100, n_steps=10000, report_step=100, reverse=True, state_size=100, words=True)
    input_size 158
    0: 4.5964474678
    1: 2.42313051224
    2: 1.8466616869
    3: 1.60127961636
    ...
    99: 0.0807262957096

words, no reverse (061e6b0ce3ed1129cd437af6b6b2653277a9cd2d)::

    $ ./autoencoder.py ~/programming/ling/corpora/opencorpora.txt \
        --n-steps=10000 --words --max-seq-length=12
    Namespace(batch_size=32, filename='/Users/kostia/programming/ling/corpora/opencorpora.txt', max_seq_length=12, min_char_count=100, n_steps=10000, report_step=100, reverse=False, state_size=100, words=True)
    input_size 158
    0: 4.61248397827
    1: 2.28991627693
    2: 1.62505555153
    3: 1.50579571724
    ...
    99: 0.136829450727

phrases (061e6b0ce3ed1129cd437af6b6b2653277a9cd2d)::

    $ ./autoencoder.py ~/programming/ling/corpora/opencorpora.txt \
        --n-steps=10000 --reverse
    Namespace(batch_size=32, filename='/Users/kostia/programming/ling/corpora/opencorpora.txt', max_seq_length=100, min_char_count=100, n_steps=10000, report_step=100, reverse=True, state_size=100, words=False)
    input_size 158
    0: 5.00039815903
    1: 2.70567774773
    2: 1.94724774361
    3: 1.91385316849
    ...
    53: 1.22337055206

phrases at 9d07f901a63695e2e7a6cc88a5c484b3ecce74d2::

    $ ./autoencoder.py ../opencorpora.txt \
        --save models/oc_ph_nor --load models/oc_ph_nor-301
    (второй прогон)
    73: loss 1.1324 in 6739 s
    oc_ph_nor-31301: loss 1.2424

    $ ./autoencoder.py ../opencorpora.txt \
        --save models/oc_ph_nor_50 --max-seq-length=50
    73: loss 1.7140 in 3374 s
    oc_ph_nor_50-63101: loss 1.0997

    $ ./autoencoder.py ~/programming/ling/corpora/opencorpora.txt \
        --save=oc_ph_nor_20 --max-seq-length=20 --load=oc_ph_nor_20-9901
    (второй прогон)
    99: loss 0.2436 in 2139 s

    $ ./autoencoder.py ~/programming/ling/corpora/opencorpora.txt \
        --max-seq-length=60 --n-layers=2 --state-size=256 --save=oc_ph_60_2_256
    136: loss 0.8219 in 29395 s
