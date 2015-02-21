# note

this code is close to 2 years old and not sure if it'll even run these days :(

# summary

quick theano implementation of <a href="http://machinelearning.wustl.edu/mlpapers/paper_files/BengioDVJ03.pdf">bengio et al's "a neural probabilistic language model" (2003)</a>

logistic softmax and hidden nodes taken pretty much verbatim from the deeplearning.net tute.

the projection layer code (though trivial) is at least something new.

for brevity trains over small set of POS tag trigrams (eg IN DT NN) so that the vocab (and hence embedding space) is very small. to get decent results need to train on a larger vocab on much more data. feature hashing works well if your vocab is huge (and, to be honest, it works well even if your vocab isn't huge due to zip's law)

# eg results

here's a simple viz of the embeddings for a couple of terms over time. days are (kinda) nearer to each other than some other words. kinda.

![embeddings](embeddings.png?raw=true "embeddings")

# use

<tt>
$ ./train_nplm.py --help
Usage: nplp [options]

neural probabilistic language model

Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit
  --trigrams=TRIGRAMS   trigram file
  --n-projection-dim=N_PROJECTION
                        num projection dimension
  --n-hidden=N_HIDDEN   num hidden nodes
  --l1-reg=L1_REG       L1 regression weight
  --l2-reg=L2_REG       L2 regression weight
  --batch-size=BATCH_SIZE
                        mini batch size
  --epochs=N_EPOCHS     epochs to run
  --learning-rate=LEARNING_RATE
                        weight update learning rate
  --seed=SEED           rng seed
  --weights-file=WEIGHTS_FILE
                        pickle model params from previous run
  --print-internal-vars
                        whether to wrap selected symbolic nodes in a
                        theano.print
  --dump-hidden-weights
                        whether to write projection/hidden/softmax layer
                        weights out to a file

$ ./train_nplm.py --epochs 50 --n-projection-dim 2

$ library(ggplot)




