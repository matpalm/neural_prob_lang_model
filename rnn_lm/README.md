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
epoch 0 min, mean, max perplexity 3.288 4.168 6.284 took 0.993 sec
epoch 1 min, mean, max perplexity 2.359 3.086 5.311 took 0.990 sec
epoch 2 min, mean, max perplexity 1.995 2.601 4.462 took 0.990 sec
epoch 3 min, mean, max perplexity 1.792 2.294 3.686 took 0.995 sec
epoch 4 min, mean, max perplexity 1.658 2.095 3.125 took 0.994 sec
epoch 5 min, mean, max perplexity 1.564 1.961 2.755 took 0.995 sec
epoch 6 min, mean, max perplexity 1.509 1.866 2.542 took 0.994 sec
epoch 7 min, mean, max perplexity 1.477 1.793 2.402 took 0.994 sec
epoch 8 min, mean, max perplexity 1.458 1.737 2.302 took 0.993 sec
epoch 9 min, mean, max perplexity 1.446 1.693 2.228 took 0.993 sec
```


