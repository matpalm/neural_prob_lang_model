# trivial neural probabilistic language model

update of NPLM to play with visualising embeddings.

see <a href="http://matpalm.com/blog/">blog</a> (still wip)

## build simple embeddings for graph with "equivalent" nodes

consider a sequence generated from a random walk of the following graph. all sequences will be of the form A? B? C? D? A? ...

![](generating_graph.png?raw=true)

```
$ ./gen_simple_data.py 1000 | ./ngrams.py 3 | shuf > trigrams.txt
$ head trigrams.txt -n 5
A2 B1 C2
A2 B2 C1
A1 B1 C1
B1 C2 D1
A2 B2 C1
```

train softmax model

```
# train simple model (cpu faster for this tiny data)
$ THEANO_FLAGS=device=cpu ./nplm_sm.py --embedding-dim=2 --n-hidden=2 --batch-size=250 --epochs=5000 --trigrams=trigrams.txt
```

plot embeddings in R

```
R>
library(ggplot2)
df = read.delim("embeddings.tsv", h=T)
# add a graph_label field, but only populated for final epoch
#df$label = as.factor(df$idx)
df$graph_label = df$label                    # copy label to graph label
df[df$iter!=max(df$iter),]$graph_label = ""  # clear label on all but last iter
# plot
ggplot(df, aes(d0, d1)) + 
  geom_path(aes(size=iter, colour=label)) +
  geom_text(aes(label=graph_label))
```

trivial 1d embedding convergence 

![](embeddings.1d.png?raw=true)

trivial 2d embedding convergence

![](embeddings.2d.png?raw=true)

## using a more language-like grammar

this time use an erdos renyi random graph as a generating grammar

```
./gen_random_data.py --seed=5 --er-n=15 --to-dot | dot -Tpng > trigram_generating_graph.png
```

![](trigram_generating_graph.png?raw=true)

```
$ ./gen_random_data.py --seed=5 --er-n=15 --gen=5
D C
A F A F A A
A A
E D C
E A A

$ ./gen_random_data.py --seed=5 --er-n=15 --gen=1000 | ./ngrams.py 3 > trigrams.txt

$ cat trigrams.txt | sush   # alias sush='sort|uniq -c|sort -nr|head'
    392 F A A
    243 A F A
    203 B F A
    199 F A F
    166 A B B
```

train softmax

```
$ ./nplm_sm.py --embedding-dim=3 --n-hidden=3 > distributions.txt
```

learns correct distributions

```
$ grep "^F A" trigrams.txt | sort | uniq -c
    392 F A A  # 66%
    199 F A F  # 33%

$ grep "w1=F,w2=A" distributions.txt | tail -n1
* P(W3|(w1=F,w2=A)=[(0.67070073, 'A'), (0.32235387, 'F'), (0.0065261312, 'B'), (0.00039717069, 'D'), (1.3933395e-05, 'E'), (8.1580602e-06, 'C')]
```

but also hallucinates (as all softmaxs do)

```
$ grep "^C C" trigrams.txt
# never occurs

$ grep "w1=C,w2=C" distributions.txt | tail -n1
  P(W3|(w1=C,w2=C)=[(0.40564349, 'B'), (0.34673718, 'A'), (0.11809964, 'E'), (0.10600764, 'D'), (0.020854352, 'F'), (0.0026576959, 'C')]
```

next train logisitic regression top layer with explicit negatives

first generate negatives

```
$ cat trigrams.txt | ./add_negative_examples.py | shuf > trigrams.with_negs.txt
$ head -n5 trigrams.with_negs.txt
A F C 1
E C E 1
B C D 1
D A D 0
B C A 0
```

then train nplm_lr.py 

WIP

