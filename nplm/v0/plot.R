library(ggplot2)
library(reshape)
df = read.delim("projection_layer_weights.1392726696.tsv")
df = df[df$epoch == max(df#epoch),]
df$epoch = NULL
df = cast(df, feature ~ node)
ggplot(df, aes(n0, n1)) + geom_point() + geom_text(aes(label=feature))