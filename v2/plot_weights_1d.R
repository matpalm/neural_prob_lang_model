library(ggplot2)
setwd("/home/mat/dev/neural_prob_lang_model/v2/")

df = read.delim("embeddings.tsv", h=T)

#df = read.delim("hidden_weights.tsv", h=T)
#df$label = as.factor(df$idx)

# add a graph_label field, but only populated for final epoch
df$graph_label = df$label
df[df$iter!=max(df$iter),]$graph_label = ""

ggplot(df, aes(iter, d0)) + 
  geom_path(aes(colour=label)) +
  geom_text(aes(label=graph_label))
