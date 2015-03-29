# trivial neural probabilistic language model

update of NPLM to play with visualising embeddings.

see <a href="http://matpalm.com/blog/2015/03/15/hallucinating_softmaxs/">http://matpalm.com/blog/2015/03/15/hallucinating_softmaxs/</a>

## using a language-like grammar

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
$ ./nplm.py --mode=sm --trigrams=trigrams.txt --embedding-dim=3 --n-hidden=3 > distributions.txt
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

then train nplm.py 

```
$ ./nplm.py --mode=lr --trigrams=trigrams.with_negs.txt --embedding-dim=3 --n-hidden=3 > distributions.txt
e 2040 b 188 cost 0.371997481934 last_batch_time 0.262334108353
1153  F C E  0.858568  # 1.0, freq 1450, rank 1/50 of pos
1153  F B A  0.584878  # 1.0, freq 16, rank 46/50
1153  C E B  0.133438  # 0.0, freq 827, rank 1/80 of negs
1153  D C D  0.78613   # 0.0, freq 14, rank 75/80
1153  F A A  0.60761   # 1.0, from nplm_sm,  FAA 60% vs FAF
1153  F A F  0.60761   # 1.0, see above
1153  C C B  0.13323   # 0.0, hallucinated in sm but more explicit no in this case.
```

over time (note: includes large y jitter)

![](nplm_lr.png?raw=true)

interesting that FAF and FAA become so tied

## real text

using 1e6 sentence data, specifically ./sentences_to_embeddables.py --emit=lemma --strip-CD --add-pos-tag --keep-top=50000

```
time cat sentences.lemma.CD.75K.ssv | ./ngrams.py 3 > sentences.lemma.CD.75K.trigrams
```

```
THEANO_FLAGS=device=gpu ./nplm.py --trigrams=/data2/1e6_sentences/sentences.lemma.CD.75K.trigrams \
 --batch-size=256 --embedding-dim=20 --n-hidden=40 --epochs=100 --adaptive-learning-rate=rmsprop \
 --cost-progress-freq=100 --dump-matrices-freq=50000 --output_file_prefix=ON2
```

## tsne projections

$ cat embeddings.tsv | grep -P "^5\t71655" | grep -P "monday|tuesday|wednesday|thursday|friday|saturday|sunday" | ./embeddings_tsne.py > embeddings.dow.2d.tsv




## older stuff, still here for images..

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
# train simple model (cpu faster for this tiny data)  NOTE: this example is deprecated and code wasn't ported, see lang stuff below
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




