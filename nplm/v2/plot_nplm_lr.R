library(ggplot2)
setwd("/home/mat/dev/neural_prob_lang_model/v2/")
df = read.delim("foo.txt", sep="\t", col.names=c("i", "key", "cost"))
df = df[df$key!='cost',]
df = df[df$i<500,]
ggplot(df, aes(i, cost)) +
 geom_point(aes(colour=key), size=2, position = position_jitter(height=0.02)) +
 theme(legend.text=element_text(size=15))
