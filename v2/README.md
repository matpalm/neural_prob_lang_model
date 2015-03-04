# trivial neural probabilistic language model

update of NPLM to play with visualising embeddings.

see <a href="http://matpalm.com/blog/">blog</a> (still wip)

## build simple embeddings

consider a sequence generated from a random walk of the following graph. all sequences will be of the form A? B? C? D? A? ...

![](generating_graph.png?raw=true)

```
# make some trivial trigram data
$ ./simple_gen_data.py 1000 | ./ngrams.py 3 | shuf > trigrams.txt
$ head trigrams.txt -n 5
A2 B1 C2
A2 B2 C1
A1 B1 C1
B1 C2 D1
A2 B2 C1
```

```
# train simple model (cpu faster for this tiny data)
$ THEANO_FLAGS=device=cpu ./simple_nplm.py --embedding-dim=1 --n-hidden=1 --batch-size=250 --epochs=5000 --trigrams=trigrams.txt
...
e 3750 b 0 i 15000
negative_log_likelihood [array(0.7252506613731384, dtype=float32)]
# actually observable transistions
P(W3|(w1=A1,w2=B1)=[(0.49844536, 'C1'), (0.48987252, 'C2'), (0.0056143967, 'D1'), (0.0053382437, 'D2')]
P(W3|(w1=B1,w2=C1)=[(0.49209341, 'D1'), (0.46864474, 'D2'), (0.00929996, 'A2'), (0.0085294973, 'C2')]
P(W3|(w1=C1,w2=D1)=[(0.52730381, 'A2'), (0.45957905, 'A1'), (0.0066987295, 'D1'), (0.0064050369, 'D2')]
P(W3|(w1=D1,w2=A1)=[(0.5231514, 'B1'), (0.47057655, 'B2'), (0.0031056409, 'D2'), (0.0030929551, 'D1')]
# Hallucinated transistions (interestingly bimodal)
P(W3|(w1=A1,w2=C1)=[(0.48817447, 'C1'), (0.47883824, 'C2'), (0.016781624, 'D1'), (0.015840771, 'D2')]
P(W3|(w1=A1,w2=D1)=[(0.43593761, 'D1'), (0.39520127, 'D2'), (0.10268536, 'A2'), (0.022749051, 'C1')]
P(W3|(w1=A1,w2=A2)=[(0.49858621, 'C1'), (0.49000439, 'C2'), (0.0054715569, 'D1'), (0.0052030799, 'D2')]
runtime 4.40494585037
```

## plot in R

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

## trivial 1d embedding convergence 

![](embeddings.1d.png?raw=true)

## trivial 2d embedding convergence

![](embeddings.2d.png?raw=true)


