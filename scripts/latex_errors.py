"""Print errors in LaTeX format."""
import argparse
import random
import re
import sys

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('error_analysis_file')
  parser.add_argument('name')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def main():
  random.seed(0)
  examples = []
  cur_lines = []
  with open(OPTS.error_analysis_file) as f:
    for line in f:
      line = line.strip()
      if line.startswith('x_orig') or line.startswith('x_pert'):
        line = re.sub(r'\x1B\[3[16]m', r'\\textbf{', line)
        line = re.sub(r'\x1B\[0m', r'}', line)
        line = re.sub('&', r'\\&', line)
        line = re.sub('x_orig', 'Original', line)
        line = re.sub('x_pert', 'Perturbed', line)
        cur_lines.append(r'\fbox{ \begin{minipage}{\textwidth}')
        cur_lines.append(line)
        cur_lines.append(r'\end{minipage} }')
      elif line.startswith('y     : '):
        num = int(line[len('y     : '):])
        label = 'positive' if num else 'negative'
        cur_lines.append(r'Correct label: %s. \\' % label)
      elif line.startswith('orig prob'):
        prob = float(line[len('orig prob  : '):])
        cur_lines.append('Model confidence on original example: %.1f.' % (prob * 100))
        cur_lines.append('\\end{figure*}')
        examples.append(cur_lines)
        cur_lines = []
  random.shuffle(examples)
  for i, ex in enumerate(examples):
    print(r'\begin{figure*}')
    print(r'%s, example %d \\' % (OPTS.name, i + 1))
    for line in ex:
      print(line)
    print('')

if __name__ == '__main__':
  OPTS = parse_args()
  main()

