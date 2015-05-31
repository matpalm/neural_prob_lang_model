import sys
import numpy as np

def perplexity_of_sequence(probabilities):
    perplexity_sum = sum([np.log2(max(1e-10, p)) for p in probabilities])
    return 2**((-1./len(probabilities)) * perplexity_sum)

def stats(values):
    summary_stats = [min(values), np.median(values), max(values)]
    return "(" + " ".join(["%.3f" % v for v in summary_stats]) + ")"

def perplexity_stats(prob_seqs):
    perplexities = [perplexity_of_sequence(prob_seq) for prob_seq in prob_seqs]
    second_last_probs = [prob_seq[-2] for prob_seq in prob_seqs]
    return "min, mean, max  perplexity %s  second_last %s" % (stats(perplexities), stats(second_last_probs))

def prob_stats(x, y, probs):
    probs_str = ["%.2f" % p for p in probs]
    return "xyp " + " ".join(map(str, zip(x, y, probs_str)))

#def pad_sequences(seqs, padding="0"):
#    max_sequence_length = max([len(seq) for seq in seqs])
#    for seq in seqs:
#        while len(seq) < max_sequence_length:
#            seq.append(padding)

