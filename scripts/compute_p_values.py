import argparse
from scipy import stats
import sys

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('scores_1')
  parser.add_argument('scores_2')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def read_scores(fn):
  with open(fn) as f:
    return [float(x) for x in f]

def main():
  x1 = read_scores(OPTS.scores_1)
  x2 = read_scores(OPTS.scores_2)
  print('Avg1 = %d/%d = %.2f%%' % (sum(x1), len(x1), 100 * sum(x1) / len(x1)))
  print('Avg2 = %d/%d = %.2f%%' % (sum(x2), len(x2), 100 * sum(x2) / len(x2)))
  s_ttest_unpaired, p_ttest_unpaired = stats.ttest_ind(x1, x2)
  print('Unpaired t-test: p=%.2e' % p_ttest_unpaired)
  s_ttest_paired, p_ttest_paired = stats.ttest_rel(x1, x2)
  print('Paired t-test: p=%.2e' % p_ttest_paired)
  s_mwu, p_mwu = stats.mannwhitneyu(x1, x2)
  print('Mann-Whitney U test (unpaired): p=%.2e' % p_mwu)
  s_wilcoxon, p_wilcoxon = stats.wilcoxon(x1, x2)
  print('Wilcoxon signed-rank test (paired): p=%.2e' % p_wilcoxon)

if __name__ == '__main__':
  OPTS = parse_args()
  main()

