# RNN language model hacking

## reber grammar

the [reber grammar](http://www.willamette.edu/~gorr/classes/cs449/reber.html) is a old standard 
for RNN testing. in particular we'll use the embedded form of the grammar.

```
$ ./generate.py --num 1000 > training
$ ./generate.py --num 100 > test
$ head -n5 test
BPBPTVVEPE
BPBTSSSXXVVEPE
BTBTSSXXVPXVPXVPXTTTTVVETE
BPBPVPSEPE
BTBPTVVETE
```

one interesting thing to note in the embedded form of the grammar is that the second token is
always the same as the second last token; either a P or a T. this is one of the long term dependencies the
model needs to learn to handle, though we'll see it's trivial for some models.

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

we'll examine a number of differing models for this task and report two stats
1) [perplexity](http://en.wikipedia.org/wiki/Perplexity#Perplexity_per_word)
and 2) the precision of predicting the second last character.

## trivial models

### sanity check models

just included as a sanity check stats

```
# just assume P(w) is uniform (grammar has 7 items; 1/7 = 0.143)
$ ./uniform_model.py training test  
min, mean, max  perplexity (7.000 7.000 7.000)  second_last (0.143 0.143 0.143)

# perfect model predicts every transistion perfectly
$ ./perfect_model.py training test  
min, mean, max  perplexity (1.000 1.000 1.000)  second_last (1.000 1.000 1.000)
```

### unigram model

P(w_{n} | w_{n-1})

not much better than just a uniform model. 
terrible at the second last prediction since it's just the frequency of the observed tokens.

```
$ ./unigram_model.py training test
min, mean, max  perplexity (5.844 6.726 8.072)  second_last (0.164 0.181 0.209)
```

### bigram model

P(w_{n} | w_{n-1}, W_{n-2})

```
$ ./bigram_model.py training test
min, mean, max  perplexity (2.742 3.128 3.933)  second_last (0.495 0.499 0.505)
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

can see a much lower perplexity compares to ngram model and second_last precision tending to 1.0

```
$ ./simple_rnn.py training test --adaptive-learning-rate=vanilla
compilation took 6.698 s
epoch 0 min, mean, max  perplexity (3.321 4.162 5.726)  second_last (0.194 0.253 0.325) took 0.994 sec
epoch 1 min, mean, max  perplexity (2.183 2.746 4.032)  second_last (0.421 0.519 0.640) took 1.004 sec
epoch 2 min, mean, max  perplexity (1.775 2.173 3.247)  second_last (0.739 0.807 0.874) took 0.994 sec
epoch 3 min, mean, max  perplexity (1.613 1.935 2.788)  second_last (0.864 0.902 0.942) took 0.995 sec
epoch 4 min, mean, max  perplexity (1.531 1.808 2.497)  second_last (0.914 0.937 0.966) took 0.995 sec
```

#### v2. using rmsprop adaptive learning rate

* same as simple but using rmsprop (the default for --adaptive-learning-rate)
* uses double the parameters as simple; each param has a stored gradient moving average

main difference to previous model is convergence much faster

```
$ ./simple_rnn.py training test
compilation took 6.55 s
epoch 0 min, mean, max  perplexity (1.338 1.540 2.209)  second_last (1.000 1.000 1.000) took 0.999 sec
epoch 1 min, mean, max  perplexity (1.370 1.538 2.141)  second_last (1.000 1.000 1.000) took 1.001 sec
epoch 2 min, mean, max  perplexity (1.368 1.537 2.107)  second_last (1.000 1.000 1.000) took 0.997 sec
epoch 3 min, mean, max  perplexity (1.354 1.558 2.292)  second_last (1.000 1.000 1.000) took 0.998 sec
epoch 4 min, mean, max  perplexity (1.359 1.558 2.290)  second_last (1.000 1.000 1.000) took 1.003 sec
```

#### v3. bidirectional rnn

* same as rmsprop version but with bidirectional layer
* uses double the parameters as rmsprop version; needs Wx, Wrec & Wy for _both_ directions

```
$ ./bidirectional_rnn.py training test
compilation took 18.921 s
epoch 0 min, mean, max  perplexity (1.087 1.268 1.885)  second_last (1.000 1.000 1.000) took 1.935 sec
epoch 1 min, mean, max  perplexity (1.086 1.263 2.148)  second_last (1.000 1.000 1.000) took 1.945 sec
epoch 2 min, mean, max  perplexity (1.073 1.291 2.172)  second_last (1.000 1.000 1.000) took 1.940 sec
epoch 3 min, mean, max  perplexity (1.073 1.308 2.504)  second_last (1.000 1.000 1.000) took 1.948 sec
epoch 4 min, mean, max  perplexity (1.073 1.282 2.942)  second_last (1.000 1.000 1.000) took 1.950 sec
```

#### v4. gru

* same as simple (unidirectional) but this time with [GRU](http://arxiv.org/abs/1502.02367)

```
$ ./gru_rnn.py training test
compilation took 11.194 s
epoch 0 min, mean, max  perplexity (1.222 1.491 3.148)  second_last (0.999 1.000 1.000) took 1.830 sec
epoch 1 min, mean, max  perplexity (1.231 1.468 1.966)  second_last (0.998 0.999 1.000) took 1.876 sec
epoch 2 min, mean, max  perplexity (1.236 1.458 1.967)  second_last (1.000 1.000 1.000) took 1.826 sec
epoch 3 min, mean, max  perplexity (1.238 1.457 2.084)  second_last (0.998 1.000 1.000) took 1.829 sec
epoch 4 min, mean, max  perplexity (1.227 1.471 2.035)  second_last (0.997 1.000 1.000) took 1.816 sec
```

## conclusions

all works, but clearly need a harder problem :/
