"""Error analysis on adversary successes."""
import argparse
import collections
import math
import re
import sys
import termcolor

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('in_file')
  parser.add_argument('--out-score-file', help='Print 0-1 loss on adversary (for p-values)')
  parser.add_argument('--quiet', '-q', action='store_true')
  parser.add_argument('--snli', action='store_true')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def count_and_frac(x, f):
  count = sum(1 for e in x if f(e))
  return count, len(x), 100 * count / len(x)

def main():
  bound_pattern = re.compile(r'.*Logit bounds: (.*) <= (.*) <= (.*), cert_correct=(.*)')
  success_pattern = re.compile(r'ADVERSARY SUCCESS on \("(.*)", ([^)]*)\): Found "(.*)" with margin (.*)')
  orig_wrong_pattern = re.compile(r'ORIGINAL PREDICTION WAS WRONG')
  failure_pattern = re.compile(r'ADVERSARY FAILURE .*')
  bounds = None
  num_ex = 1
  diffs = []
  lens = []
  diff_percs = []
  probs = []
  adv_scores = []
  with open(OPTS.in_file) as f:
    for line in f:
      m_bound = re.match(bound_pattern, line.strip())
      m_success = re.match(success_pattern, line.strip())
      m_orig = re.match(orig_wrong_pattern, line.strip())
      m_failure = re.match(failure_pattern, line.strip())
      if line.startswith('ADVERSARY SUCCESS') and not m_success:
        print(line)
      if m_bound:
        cur_bounds = (float(m_bound.group(1)), float(m_bound.group(2)), 
                      float(m_bound.group(3)))
      elif m_success:
        orig_toks = m_success.group(1).split(' ')
        if OPTS.snli:
          y = m_success.group(2)
        else:
          y = int(m_success.group(2))
        perturbed_toks = m_success.group(3).split(' ')
        margin = float(m_success.group(4))
        orig_colored = [termcolor.colored(w1, 'cyan') if w1 != w2 else w1
                        for w1, w2 in zip(orig_toks, perturbed_toks)]
        perturbed_colored = [termcolor.colored(w2, 'red') if w1 != w2 else w2
                             for w1, w2 in zip(orig_toks, perturbed_toks)]
        num_diff = sum(1 for w1, w2 in zip(orig_toks, perturbed_toks) if w1 != w2)
        diff_perc = 100.0 * num_diff / len(orig_toks)
        if not OPTS.snli:
          orig_prob = 1 / (1 + math.exp(-(2 * y - 1) * cur_bounds[1]))
        if not OPTS.quiet:
          print('Case %d' % num_ex) 
          print('  x_orig: %s' % ' '.join(orig_colored))
          print('  x_pert: %s' % ' '.join(perturbed_colored))
          print('  y     : %d' % y)
          print('  diff  : %d' % num_diff)
          print('  len   : %d' % len(orig_toks))
          print('  diff %%: %.2f%%' % diff_perc)
          if not OPTS.snli:
            print('  orig prob  : %.6f' % orig_prob)
          print('  orig logits: %.6f <= %.6f <= %.6f' % cur_bounds)
          print('  new margin : %.2f' % margin)
          print()
        num_ex += 1
        diffs.append(num_diff)
        lens.append(len(orig_toks))
        diff_percs.append(diff_perc)
        if not OPTS.snli:
          probs.append(orig_prob)
        adv_scores.append(0)
      elif m_orig:
        adv_scores.append(0)
      elif m_failure:
        adv_scores.append(1)
    print('Adversarial accuracy: %d/%d = %.2f%%' % (
        sum(adv_scores), len(adv_scores), 100 * sum(adv_scores) / len(adv_scores)))
    print('Overall averages')
    print('  diff  : %.2f' % (sum(diffs) / len(diffs)))
    print('  len   : %.2f' % (sum(lens) / len(lens)))
    print('  diff %%: %.2f%%' % (sum(diff_percs) / len(diff_percs)))
    if not OPTS.snli:
      print('  probs  : %.6f' % (sum(probs) / len(probs)))
      print('  p > 0.9: %d/%d = %.2f%%' % count_and_frac(probs, lambda p: p > 0.9))
      print('  p > 0.8: %d/%d = %.2f%%' % count_and_frac(probs, lambda p: p > 0.8))
      print('  p > 0.7: %d/%d = %.2f%%' % count_and_frac(probs, lambda p: p > 0.7))
      print('  p > 0.6: %d/%d = %.2f%%' % count_and_frac(probs, lambda p: p > 0.6))
    print('  diff <= 3 : %d/%d = %.2f%%' % count_and_frac(diffs, lambda d: d <= 3))
    print('  diff >= 10: %d/%d = %.2f%%' % count_and_frac(diffs, lambda d: d >= 10))
    print('  diff histogram:')
    diff_histo = collections.Counter(diffs)
    for k in sorted(diff_histo):
      print('    %02d: %d' % (k, diff_histo[k]))
    print('  histogram in list form: [%s]' % [diff_histo[i] for i in range(1, 31)])
    if OPTS.out_score_file:
      with open(OPTS.out_score_file, 'w') as f:
        for s in adv_scores:
          print(s, file=f)

if __name__ == '__main__':
  OPTS = parse_args()
  main()

