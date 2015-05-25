library(ggplot2)

dfv = read.delim("vanilla.cost", h=F)
dfv$run = 'vanilla'
dfv$n = 1:nrow(dfv)

dfr = read.delim("rmsprop.cost", h=F)
dfr$run = 'rmsprop'
dfr$n = 1:nrow(dfr)

df = rbind(dfv, dfr)
ggplot(df, aes(n, V2)) + 
  geom_point(aes(colour=run)) + 
  geom_smooth(aes(colour=run))
