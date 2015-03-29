library(ggplot2)
setwd("/home/mat/dev/neural_prob_lang_model/v2/")
  
df = read.delim("embeddings.dow.2d.tsv", h=F)

ggplot(df, aes(d0, d1)) +
  geom_point() + 
  geom_text(aes(label=graph_label))
