library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)
input_file = paste("nplm_", args[1], ".", args[2], ".", args[3], sep="")
output_png = paste("nplm_", args[1], ".", args[2], ".", args[3], ".png", sep="")

df = read.delim(input_file, sep="\t", col.names=c("i", "w1w2w3", "w1w2", "cost"))

df = df[df$w1w2w3!='cost',]
#df = df[df$i<600,]

png(output_png, width=1000, height=450)
ggplot(df, aes(i, cost)) +
 geom_point(aes(colour=w1w2w3), alpha=0.8, size=2, position = position_jitter(height=0.00)) +
 theme(legend.text=element_text(size=15)) + facet_grid(. ~ w1w2) + ylim(0, 1) +
 ggtitle(output_png)
dev.off()
