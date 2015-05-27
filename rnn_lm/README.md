# RNN language model hacking

## reber grammar

the [reber grammar](http://www.willamette.edu/~gorr/classes/cs449/reber.html) is a old standard 
for RNN testing. in particular we'll use the embedded form of the grammar.

```
$ ./generate.py --num 1000 > training
$ ./generate.py --num 100 > test
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
$ ./generate.py --num 100000 | perl -ne'print length($_)."\n";' | histogram.py
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

### some rnns

![cost](cost.png?raw=true "cost")

#### v1. simple as you can get

* single layer RNN
* no gating within unit at all
* no adaptive learning rates / schedules, just fixed rate
* no batching, train one example at a time.
* trivial randn weight init
* no bias with dot products

```
$ ./simple_rnn.py --adaptive-learning-rate=vanilla training test
compilation took 6.698 s
epoch 0 min, mean, max perplexity 1.359 1.537 2.025 took 1.003 sec
epoch 1 min, mean, max perplexity 1.350 1.527 2.019 took 0.999 sec
epoch 2 min, mean, max perplexity 1.350 1.529 2.099 took 1.003 sec
epoch 3 min, mean, max perplexity 1.349 1.524 2.027 took 1.004 sec
epoch 4 min, mean, max perplexity 1.334 1.529 1.999 took 1.007 sec
```

#### v2. using rmsprop adaptive learning rate

* same as simple but using rmsprop (the default for --adaptive-learning-rate)
* uses double the parameters as simple; each param has a stored gradient moving average

```
$ ./simple_rnn.py training test
compilation took 6.55 s
epoch 0 min, mean, max perplexity 1.327 1.526 1.952 took 1.018 sec
epoch 1 min, mean, max perplexity 1.357 1.518 1.996 took 1.008 sec
epoch 2 min, mean, max perplexity 1.338 1.523 1.869 took 1.025 sec
epoch 3 min, mean, max perplexity 1.346 1.520 1.989 took 1.022 sec
epoch 4 min, mean, max perplexity 1.333 1.522 2.044 took 1.009 sec
```

#### v3. bidirectional rnn

* same as rmsprop version but with bidirectional layer
* uses double the parameters as rmsprop version; needs Wx, Wrec & Wy for _both_ directions

```
$ ./bidirectional_rnn.py training test
compilation took 18.921 s
epoch 0 min, mean, max perplexity 1.081 1.290 1.883 took 1.940 sec
epoch 1 min, mean, max perplexity 1.070 1.270 1.938 took 1.950 sec
epoch 2 min, mean, max perplexity 1.074 1.271 2.271 took 1.955 sec
epoch 3 min, mean, max perplexity 1.075 1.269 2.361 took 1.947 sec
epoch 4 min, mean, max perplexity 1.087 1.268 2.369 took 1.952 sec
```

#### v4. gru

* unidirectional but this time with [GRU](http://arxiv.org/abs/1502.02367)

```
$ ./gru_rnn.py training test
compilation took 11.194 s
epoch 0 min, mean, max perplexity 1.233 1.464 1.969 took 1.792 sec
epoch 1 min, mean, max perplexity 1.271 1.449 1.904 took 1.794 sec
epoch 2 min, mean, max perplexity 1.272 1.444 1.911 took 1.794 sec
epoch 3 min, mean, max perplexity 1.277 1.446 1.999 took 1.818 sec
epoch 4 min, mean, max perplexity 1.264 1.450 1.974 took 1.810 sec
```

