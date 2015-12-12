LSTM Character Autoencoder
==========================

Note that loss is NOT perplexity.

words with reverse::

    $ python autoencoder.py ~/programming/ling/corpora/opencorpora.txt \
        --n-steps=10000 --reverse --words --max-seq-length=12
    reading inputs
    input_size 158
    I tensorflow/core/common_runtime/local_device.cc:40] Local device intra op parallelism threads: 4
    I tensorflow/core/common_runtime/direct_session.cc:60] Direct session inter op parallelism threads: 4
    0: 4.5964474678
    1: 2.42313051224
    2: 1.8466616869
    3: 1.60127961636
    ...
    99: 0.0807262957096

words, no reverse::

    $ python autoencoder.py ~/programming/ling/corpora/opencorpora.txt \
        --n-steps=10000 --words --max-seq-length=12
    reading inputs
    input_size 158
    0: 4.61248397827
    1: 2.28991627693
    2: 1.62505555153
    3: 1.50579571724
    ...
    99: 0.136829450727
