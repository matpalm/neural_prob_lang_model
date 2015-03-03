# trivial neural probabilistic language model

update of NPLM to play with visualising embeddings.

see <a href="http://matpalm.com/blog/">blog</a> (still wip)

## build simple embeddings

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
P(W3|(w1=A1,w2=B1)=[(0.48999223, 'C1'), (0.47966138, 'C2'), (0.010732187, 'A1'), (0.007289567, 'D2')]
P(W3|(w1=B1,w2=C1)=[(0.494946, 'D1'), (0.47194412, 'D2'), (0.012410824, 'B1'), (0.010464143, 'B2')]
P(W3|(w1=C1,w2=D1)=[(0.51963907, 'A2'), (0.44903326, 'A1'), (0.011842941, 'C2'), (0.007554505, 'B2')]
P(W3|(w1=D1,w2=A1)=[(0.50821102, 'B1'), (0.45581818, 'B2'), (0.011167618, 'D1'), (0.0083318818, 'A1')]
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


