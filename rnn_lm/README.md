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

```
$ ./uniform_model.py training test
min, mean, max perplexity 7.0 7.0 7.0
```

```
$ ./unigram_model.py training test
min, mean, max perplexity 5.65320988827 6.58426607105 8.0107271951
```

```
$ ./bigram_model.py training test
min, mean, max perplexity 3.18062061871 3.54441314096 4.09765369567
```

```
# just for sanity...
$ ./perfect_model.py training test
min, mean, max perplexity 1.0 1.0 1.0
```

