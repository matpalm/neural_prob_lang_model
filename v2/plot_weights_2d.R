library(ggplot2)
df = read.delim("/data/nplm/exp8/cost.1427953157.tsv", h=T)
ggplot(df, aes(time, cost)) + geom_point() + geom_smooth() + ylim(2, 4)