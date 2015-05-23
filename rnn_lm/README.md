# RNN language model hacking

## reber grammar

the [reber grammar](http://www.willamette.edu/~gorr/classes/cs449/reber.html) is a old standard 
for RNN testing. in particular we'll use the embedded form of the grammar.

```
$ ./generate.py 1000 > training
$ ./generate.py 100 > test
$ head -n5 test
BTBTXXTVPSETE
BTBPVVETE
BTBPVPXTTTVPSETE
BTBTXXVVETE
BTBTSSXSETE
```

lengths of string are potentially unbounded but majority are <20
(histogram.py provided by the awesome [data_hacks](https://github.com/bitly/data_hacks) lib)

```
$ ./generate.py 100000 | perl -ne'print length($_)."\n";' | histogram.py 
# NumSamples = 100000; Min = 10.00; Max = 45.00
# Mean = 13.007970; Variance = 11.508306; SD = 3.392389; Median 12.000000
# each ∎ represents a count of 896
   10.0000 -    13.5000 [ 67254]: ∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎
   13.5000 -    17.0000 [ 22887]: ∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎
   17.0000 -    20.5000 [  5923]: ∎∎∎∎∎∎
   20.5000 -    24.0000 [  2766]: ∎∎∎
   24.0000 -    27.5000 [   702]: 
   27.5000 -    31.0000 [   325]: 
   31.0000 -    34.5000 [    93]: 
   34.5000 -    38.0000 [    37]: 
   38.0000 -    41.5000 [     8]: 
   41.5000 -    45.0000 [     5]: 
```

## trivial models

### sanity check models

mainly included as a sanity check of perplexity range

```
# just assume P(w) is uniform (grammar has 7 items)
$ ./uniform_model.py training test  
min, mean, max perplexity 7.0 7.0 7.0

# perfect model predicts every transistion perfectly
$ ./perfect_model.py training test  
min, mean, max perplexity 1.0 1.0 1.0
```

### unigram model

P(w_{n} | w_{n-1})

```
$ ./unigram_model.py training test
min, mean, max perplexity 5.65320988827 6.58426607105 8.0107271951
```

### bigram model

P(w_{n} | w_{n-1}, W_{n-2})

```
$ ./bigram_model.py training test
min, mean, max perplexity 3.18062061871 3.54441314096 4.09765369567
```

### super dead simple rnn

simple as you can RNN.

* no gating within unit at all
* no adaptive learning rates / schedules, just fixed rate
* no batching, train one example at a time.
* trivial randn weight init

```
$ ./simple_rnn_model.py training test
epoch 0 min, mean, max perplexity 2.569 3.433 5.408 took 0.998 sec
epoch 1 min, mean, max perplexity 2.029 2.652 4.381 took 1.006 sec
epoch 2 min, mean, max perplexity 1.826 2.341 3.825 took 1.010 sec
epoch 3 min, mean, max perplexity 1.734 2.155 3.451 took 1.008 sec
epoch 4 min, mean, max perplexity 1.693 2.030 3.171 took 1.007 sec
epoch 5 min, mean, max perplexity 1.648 1.938 2.948 took 1.009 sec
epoch 6 min, mean, max perplexity 1.614 1.862 2.774 took 1.009 sec
epoch 7 min, mean, max perplexity 1.566 1.792 2.627 took 1.012 sec
epoch 8 min, mean, max perplexity 1.508 1.732 2.474 took 1.010 sec
epoch 9 min, mean, max perplexity 1.470 1.686 2.338 took 1.010 sec
```


