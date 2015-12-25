Results
=======

.. contents::

Note that loss is NOT perplexity.

words with reverse
------------------

061e6b0ce3ed1129cd437af6b6b2653277a9cd2d

::

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


words, no reverse
-----------------

061e6b0ce3ed1129cd437af6b6b2653277a9cd2d

::

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

phrases
-------

061e6b0ce3ed1129cd437af6b6b2653277a9cd2d

::

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

phrases
-------

27e9ce95f43aabacf2ca97c83d36805820ee77ba

::

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

Learning gone bad::

    $ ./autoencoder.py ~/programming/ling/corpora/opencorpora.txt \
        --max-seq-length=60 --n-layers=2 --state-size=256 --save=oc_ph_60_2_256
    136: loss 0.8219 in 29395 s
    ...
    775: loss 0.1490 in 133614 s
    ...
    809: loss 0.1506 in 139551 s
    810: loss 0.9073 in 139726 s
    811: loss 3.2113 in 139898 s



::

    $ ./autoencoder.py --state-size=100 --n-layers=2 ../corpora/opencorpora.txt --save=models/oc_100_2
    Namespace(batch_size=64, evaluate=False, filename='../corpora/opencorpora.txt', load=None, max_seq_length=60, min_char_count=100, n_layers=2, n_steps=100000, predict=False, report_step=100, reverse=False, save='models/oc_100_2', state_size=100, words=False)
    158 chars, train set size 92112, valid set size 2000
        1: train loss 4.9637, valid loss 4.9021 in 12 s
      100: train loss 3.1555, valid loss 2.7622 in 157 s
      200: train loss 2.7090, valid loss 2.6445 in 307 s
      300: train loss 2.5380, valid loss 2.4422 in 445 s
      400: train loss 2.3865, valid loss 2.3276 in 582 s
      500: train loss 2.3022, valid loss 2.2505 in 721 s
      600: train loss 2.2390, valid loss 2.1890 in 857 s

::

    $ ./autoencoder.py --state-size=100 --n-layers=2 easy-5k-mean.txt --save=models/easy_100_2
    Namespace(batch_size=64, evaluate=False, filename='easy-5k-mean.txt', load=None, max_seq_length=60, min_char_count=100, n_layers=2, n_steps=100000, predict=False, report_step=100, reverse=False, save='models/easy_100_2', state_size=100, words=False)
    87 chars, train set size 4500, valid set size 500
        1: train loss 4.3770, valid loss 4.3390 in 5 s
      100: train loss 2.8475, valid loss 2.5244 in 110 s
      200: train loss 2.5042, valid loss 2.4364 in 216 s
      300: train loss 2.3494, valid loss 2.2305 in 322 s
      400: train loss 2.1734, valid loss 2.0840 in 427 s
      500: train loss 2.0741, valid loss 2.0237 in 540 s
      600: train loss 2.0151, valid loss 1.9727 in 650 s
      700: train loss 1.9562, valid loss 1.9334 in 759 s
      800: train loss 1.9228, valid loss 1.9041 in 871 s
      900: train loss 1.9010, valid loss 1.8723 in 984 s
     1000: train loss 1.8661, valid loss 1.8517 in 1096 s
     1100: train loss 1.8402, valid loss 1.8177 in 1205 s
     1200: train loss 1.8362, valid loss 1.8067 in 1314 s


::

    $ ./autoencoder.py ../easy-5k-mean.txt --state-size=256 --n-layers=2 --save models/easy_256_2 --max-seq-length 100
    Namespace(batch_size=64, evaluate=False, filename='../easy-5k-mean.txt', load=None, max_seq_length=100, min_char_count=100, n_layers=2, n_steps=100000, predict=False, report_step
    =100, reverse=False, save='models/easy_256_2', state_size=256, words=False)
    87 chars, train set size 4500, valid set size 500
        1: train loss 4.4241, valid loss 4.2667 in 11 s
      100: train loss 2.2933, valid loss 2.0396 in 174 s
      200: train loss 1.9721, valid loss 1.8683 in 336 s
      300: train loss 1.7892, valid loss 1.7046 in 498 s
      400: train loss 1.6760, valid loss 1.6166 in 659 s
      500: train loss 1.5971, valid loss 1.5617 in 819 s
      600: train loss 1.5493, valid loss 1.5217 in 980 s
      700: train loss 1.4865, valid loss 1.4815 in 1141 s
      800: train loss 1.4489, valid loss 1.4511 in 1302 s
      900: train loss 1.4336, valid loss 1.4294 in 1463 s
     1000: train loss 1.3983, valid loss 1.4019 in 1625 s
     1100: train loss 1.3877, valid loss 1.3807 in 1786 s
     1200: train loss 1.3583, valid loss 1.3598 in 1947 s


::

    ./autoencoder.py ../opencorpora.txt --state-size=256 --n-layers=2 --save models/oc_256_2 --max-seq-length 100 --report-step=200
    Namespace(batch_size=64, evaluate=False, filename='../opencorpora.txt', load=None, max_seq_length=100, min_char_count=100, n_layers=2, n_steps=100000, predict=False, report_step=200, reverse=False, save='models/oc_256_2', state_size=256, words=False)
    158 chars, train set size 93112, valid set size 1000
        1: train loss 5.0125, valid loss 4.9021 in 12 s
      200: train loss 2.6352, valid loss 2.3482 in 338 s


random-limit
------------

random-limit speeds up training a lot (028ff8422a0dd7fcd451d3dcf78d0b7c226eb4dc)::


    ./autoencoder.py --state-size=256 frank.txt --max-seq-length=40 --random-limit --min-char-count 10 --save models/frank_256_1_40_rand 
    Namespace(batch_size=64, evaluate=False, filename='frank.txt', load=None, max_gradient_norm=5.0, max_seq_length=40, min_char_count=10, n_layers=1, n_steps=100000, predict=False, random_limit=True, report_step=100, reverse=False, save='models/frank_256_1_40_rand', state_size=256, words=False)
    71 chars, train set size 1116, valid set size 125
        1: train loss 4.2112, valid loss 4.1370 in 1 s
      100: train loss 1.8699, valid loss 1.6533 in 50 s
      200: train loss 1.3919, valid loss 1.5078 in 100 s
      300: train loss 1.3081, valid loss 1.4206 in 149 s
      400: train loss 1.2193, valid loss 1.3295 in 197 s
      500: train loss 1.1664, valid loss 1.2877 in 247 s
      600: train loss 1.0852, valid loss 1.2488 in 298 s
      700: train loss 1.0703, valid loss 1.2186 in 347 s
      800: train loss 1.0545, valid loss 1.2320 in 395 s
      900: train loss 1.0142, valid loss 1.1727 in 443 s
     1000: train loss 1.0199, valid loss 1.1898 in 492 s
     1100: train loss 0.9755, valid loss 1.1397 in 541 s
     1200: train loss 0.9711, valid loss 1.1185 in 592 s
     1300: train loss 0.9422, valid loss 1.1158 in 657 s
     1400: train loss 0.9245, valid loss 1.0925 in 724 s
     1500: train loss 0.9045, valid loss 1.0827 in 774 s
     1600: train loss 0.8787, valid loss 1.0723 in 825 s
     1700: train loss 0.8769, valid loss 1.0493 in 877 s
     1800: train loss 0.8309, valid loss 1.0453 in 926 s
     1900: train loss 0.8317, valid loss 1.0446 in 975 s
     2000: train loss 0.8111, valid loss 1.0243 in 1028 s
     2100: train loss 0.7998, valid loss 1.0261 in 1080 s
     2200: train loss 0.7740, valid loss 1.0078 in 1133 s
     2300: train loss 0.7568, valid loss 1.0014 in 1184 s
     2400: train loss 0.7449, valid loss 0.9908 in 1233 s

same test on opencorpora - "harder" corpora does not matter much::

    $ ./autoencoder.py --state-size=256 ../corpora/opencorpora.txt --max-seq-length=40 --random-limit --save models/oc_256_1_40_rand
    Namespace(batch_size=64, evaluate=False, filename='../corpora/opencorpora.txt', load=None, max_gradient_norm=5.0, max_seq_length=40, min_char_count=100, n_layers=1, n_steps=100000, predict=False, random_limit=True, report_step=100, reverse=False, save='models/oc_256_1_40_rand', state_size=256, words=False)
    158 chars, train set size 92112, valid set size 2000
        1: train loss 4.9348, valid loss 4.8035 in 7 s
      100: train loss 1.9603, valid loss 1.5191 in 84 s
      200: train loss 1.4415, valid loss 1.3879 in 162 s
      300: train loss 1.3460, valid loss 1.2915 in 241 s
      400: train loss 1.2405, valid loss 1.2089 in 319 s
      500: train loss 1.1727, valid loss 1.1614 in 426 s
      600: train loss 1.1233, valid loss 1.0886 in 501 s
      700: train loss 1.0895, valid loss 1.0604 in 575 s
      800: train loss 1.0549, valid loss 1.0389 in 648 s
      900: train loss 1.0435, valid loss 1.0111 in 721 s
     1000: train loss 1.0261, valid loss 1.0093 in 794 s
     1100: train loss 0.9910, valid loss 0.9871 in 870 s

2 layers::

     $ ./autoencoder.py --state-size=256 --n-layers=2 ../corpora/opencorpora.txt --max-seq-length=60 --random-limit --save models/oc_256_2_60_rand
        1: train loss 4.9641, valid loss 4.6453 in 21 s
      100: train loss 1.7707, valid loss 1.4325 in 270 s
      200: train loss 1.3694, valid loss 1.3340 in 523 s
      ...
     3600: train loss 0.5975, valid loss 0.5989 in 8845 s

2 larger layers on AWS GPU::

    $ ./autoencoder.py --state-size=512 --n-layers=2 opencorpora.txt --max-seq-length=60 --random-limit --save models/oc_512_2_60_rand
        1: train loss 4.9794, valid loss 4.5997 in 15 s
      100: train loss 1.6813, valid loss 1.3991 in 110 s
      200: train loss 1.3115, valid loss 1.2470 in 204 s
       ...
     5300: train loss 0.2590, valid loss 0.2558 in 4908 s
     5400: train loss 0.2307, valid loss 0.2460 in 5001 s
       ...
     8200: train loss 0.1362, valid loss 0.1441 in 2032 s

   **TODO - go to convergence**

2 larger layers on AWS GPU::

    $ ./autoencoder.py --state-size=1024 --n-layers=2 opencorpora.txt --max-seq-length=60 --random-limit --save models/oc_1024_2_60_
    rand --report-step 200
    Namespace(batch_size=64, evaluate=False, filename='opencorpora.txt', load=None, max_gradient_norm=5.0, max_seq_length=60, min_char_count=100, n_layers=2, n_steps=100000, predict=
    False, random_limit=True, report_step=200, reverse=False, save='models/oc_1024_2_60_rand', state_size=1024, words=False)
    158 chars, train set size 92112, valid set size 2000
        1: train loss 5.0210, valid loss 4.5363 in 9 s
      200: train loss 1.4709, valid loss 1.2170 in 211 s
      400: train loss 1.1323, valid loss 1.0838 in 413 s
      ...
    32000: train loss 0.0128, valid loss 0.0192 in 32624 s

GRU & LSTM-basic
----------------

f2921e50065173636d5e64fa52866be5e43a114d, GRU::

    $ ./autoencoder.py --state-size=256 --n-layers=2 ../corpora/opencorpora.txt --max-seq-length=60 --random-limit --cell=gru --save models/oc_256_2_60_rand_gru
    Namespace(batch_size=64, cell='gru', evaluate=False, filename='../corpora/opencorpora.txt', load=None, max_gradient_norm=5.0, max_seq_length=60, min_char_count=100, n_layers=2, n_steps=100000, predict=False, random_limit=True, report_step=100, reverse=False, save='models/oc_256_2_60_rand_gru', state_size=256, words=False)
    158 chars, train set size 92112, valid set size 2000
        1: train loss 4.9506, valid loss 4.4924 in 23 s
      100: train loss 1.4869, valid loss 1.2472 in 319 s
      200: train loss 1.1386, valid loss 1.1019 in 615 s
      300: train loss 1.0402, valid loss 1.0187 in 897 s
      ...
     2000: train loss 0.4305, valid loss 0.4274 in 5600 s

LSTM-basic is slower, but looks better than default LSTM::

    $ ./autoencoder.py --state-size=256 --n-layers=2 ../corpora/opencorpora.txt --max-seq-length=60 --random-limit --cell=lstm-basic --save models/oc_256_2_60_rand_lstmbasic
    Namespace(batch_size=64, cell='lstm-basic', evaluate=False, filename='../corpora/opencorpora.txt', load=None, max_gradient_norm=5.0, max_seq_length=60, min_char_count=100, n_layers=2, n_steps=100000, predict=False, random_limit=True, report_step=100, reverse=False, save='models/oc_256_2_60_rand_lstmbasic', state_size=256, words=False)
    158 chars, train set size 92112, valid set size 2000
        1: train loss 4.9580, valid loss 4.5178 in 37 s
      100: train loss 1.5467, valid loss 1.2869 in 384 s
      200: train loss 1.1840, valid loss 1.1549 in 719 s
      300: train loss 1.1181, valid loss 1.1116 in 1102 s

