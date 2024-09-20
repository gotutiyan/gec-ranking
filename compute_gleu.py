import math
from collections import Counter
from typing import List, Tuple
import argparse
import sys
import os
import scipy.stats
import numpy as np
import random

class GLEU:

    def __init__(self, n: int=4) -> None:
        self.order = n

    def load_hypothesis(self, hypothesis: str) -> None:
        self.hlen = len(hypothesis.split())
        self.this_h_ngrams = [self.get_ngram_counts(hypothesis.split(), n)
                               for n in range(1, self.order+1)]

    def load_sources(self, sources: List[str]) -> None:
        self.all_s_ngrams = [[self.get_ngram_counts(line.split(), n)
                                for n in range(1, self.order+1)]
                              for line in sources]

    def load_references(self, references: List[List[str]]) -> None:
        self.refs = [[] for i in range(len(self.all_s_ngrams))]
        self.rlens = [[] for i in range(len(self.all_s_ngrams))]
        for refs in references:
            for i, line in enumerate(refs):
                self.refs[i].append(line.split())
                self.rlens[i].append(len(line.split()))

        # count number of references each n-gram appear sin
        self.all_rngrams_freq = [Counter() for i in range(self.order)]

        self.all_r_ngrams = []
        for refset in self.refs:
            all_ngrams = []
            self.all_r_ngrams.append(all_ngrams)

            for n in range(1, self.order + 1):
                ngrams = self.get_ngram_counts(refset[0], n)
                all_ngrams.append(ngrams)

                for k in ngrams.keys():
                    self.all_rngrams_freq[n-1][k] += 1

                for ref in refset[1:]:
                    new_ngrams = self.get_ngram_counts(ref, n)
                    for nn in new_ngrams.elements():
                        if new_ngrams[nn] > ngrams.get(nn, 0):
                            ngrams[nn] = new_ngrams[nn]

    def get_ngram_counts(self, sentence: List[str], n: int) -> Counter:
        return Counter([tuple(sentence[i:i+n])
                        for i in range(len(sentence) + 1 - n)])

    # returns ngrams in a but not in b
    def get_ngram_diff(self, a: Counter, b: Counter) -> Counter:
        diff = Counter(a)
        for k in (set(a) & set(b)):
            del diff[k]
        return diff

    def normalization(self, ngram: Tuple[str], n: int) -> float:
        return 1.0 * self.all_rngrams_freq[n-1][ngram] / len(self.rlens[0])

    # Collect BLEU-relevant statistics for a single hypothesis/reference pair.
    # Return value is a generator yielding:
    # (c, r, numerator1, denominator1, ... numerator4, denominator4)
    # Summing the columns across calls to this function on an entire corpus
    # will produce a vector of statistics that can be used to compute GLEU
    def gleu_stats(self, i: int, r_ind: int=None) -> int:
      hlen = self.hlen
      rlen = self.rlens[i][r_ind]
      
      yield hlen
      yield rlen

      for n in range(1, self.order + 1):
        h_ngrams = self.this_h_ngrams[n-1]
        s_ngrams = self.all_s_ngrams[i][n-1]
        r_ngrams = self.get_ngram_counts(self.refs[i][r_ind], n)

        s_ngram_diff = self.get_ngram_diff(s_ngrams, r_ngrams)

        yield max([sum((h_ngrams & r_ngrams).values()) - \
                    sum((h_ngrams & s_ngram_diff).values()), 0])

        yield max([hlen+1-n, 0])

    # Compute GLEU from collected statistics obtained by call(s) to gleu_stats
    def gleu(self, stats: List[int], smooth: bool=False) -> float:
        # smooth 0 counts for sentence-level scores
        if smooth:
            stats = [s if s != 0 else 1 for s in stats]
        if len(list(filter(lambda x: x==0, stats))) > 0:
            return 0
        (c, r) = stats[:2]
        log_gleu_prec = sum([math.log(float(x)/y)
                             for x, y in zip(stats[2::2], stats[3::2])]) / 4
        return math.exp(min([0, 1-float(r)/c]) + log_gleu_prec)

def get_gleu_stats(scores: List[float]) -> List[str]:
    mean = np.mean(scores)
    std = np.std(scores)
    ci = scipy.stats.norm.interval(0.95,loc=mean,scale=std)
    return ['%f'%mean,
            '%f'%std,
            '(%.3f,%.3f)'%(ci[0],ci[1])]

def calc_gleu(
    sources: List[str],
    hypothesis: List[str],
    references: List[List[str]],
    n=4,
    iter=500
):
    num_iterations = iter

    # if there is only one reference, just do one iteration
    if len(references) == 1:
        num_iterations = 1

    gleu_calculator = GLEU(n)

    gleu_calculator.load_sources(sources)
    gleu_calculator.load_references(references)

    for hyp in [hypothesis]:
        # first generate a random list of indices, using a different seed
        # for each iteration
        indices = []
        for j in range(num_iterations):
            random.seed(j*101)
            indices.append([random.randint(0, len(references)-1)
                            for i in range(len(hyp))])

        iter_stats = [[0 for i in range(2*n+2)]
                       for j in range(num_iterations)]

        for i, h in enumerate(hyp):

            gleu_calculator.load_hypothesis(h)
            # we are going to store the score of this sentence for each ref
            # so we don't have to recalculate them 500 times

            stats_by_ref = [None for r in range(len(references))]

            for j in range(num_iterations):
                ref = indices[j][i]
                this_stats = stats_by_ref[ref]

                if this_stats is None:
                    this_stats = [s for s in gleu_calculator.gleu_stats(
                        i, r_ind=ref)]
                    stats_by_ref[ref] = this_stats

                iter_stats[j] = [sum(scores)
                                  for scores in zip(iter_stats[j], this_stats)]
        score = get_gleu_stats([gleu_calculator.gleu(stats)
                                for stats in iter_stats])[0]
        return score

    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--reference",
                        help="Target language reference sentences. Multiple "
                        "files for multiple references.",
                        nargs="*",
                        dest="reference",
                        required=True)
    parser.add_argument("-s", "--source",
                        help="Source language source sentences",
                        dest="source",
                        required=True)
    parser.add_argument("-o", "--hypothesis",
                        help="Target language hypothesis sentences to evaluate "
                        "(can be more than one file--the GLEU score of each "
                        "file will be output separately). Use '-o -' to read "
                        "hypotheses from stdin.",
                        dest="hypothesis",
                        required=True)
    parser.add_argument("-n",
                        help="Maximum order of ngrams",
                        type=int,
                        default=4)
    parser.add_argument("-d","--debug",
                        help="Debug; print sentence-level scores",
                        default=False,
                        action="store_true")
    parser.add_argument('--iter',
                        type=int,
                        default=500,
                        help='the number of iterations to run')

    args = parser.parse_args()

    srcs = open(args.source).read().rstrip().split('\n')
    hyps = open(args.hypothesis).read().rstrip().split('\n')
    refs = [open(r).read().rstrip().split('\n') for r in args.reference]
    score = calc_gleu(
        sources=srcs,
        hypothesis=hyps,
        references=refs,
        n=args.n,
        iter=args.iter
    )
    print(score)

    
