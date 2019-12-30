"""Split IMDB based on movies."""
import argparse
import collections
import glob
import os
import random
import sys

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('imdb_dir')
  parser.add_argument('out_prefix')
  parser.add_argument('-s', '--rng_seed', type=int, default=314159)
  parser.add_argument('-f', '--train-frac', type=float, default=0.8)
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def read_urls(label):
  with open(os.path.join(OPTS.imdb_dir, 'train', 'urls_%s.txt' % label)) as f:
    return [line.strip() for line in f]

def read_files(label):
  files = glob.glob(os.path.join(OPTS.imdb_dir, 'train', label, '*.txt'))
  files.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
  return files

def add_data(urls, files, url_to_file):
  for u, f in zip(urls, files):
    url_to_file[u].append(f)

def write_data(split, files):
  out_fn = os.path.join(OPTS.imdb_dir, 'train', '%s_%s_files.txt' % (OPTS.out_prefix, split))
  with open(out_fn, 'w') as f:
    for fn in files:
      prefix_len = len(os.path.join(OPTS.imdb_dir, 'train')) + 1 
      print(fn[prefix_len:], file=f)

def main():
  random.seed(OPTS.rng_seed)
  pos_urls = read_urls('pos')
  neg_urls = read_urls('neg')
  pos_files = read_files('pos')
  neg_files = read_files('neg')
  url_to_file = collections.defaultdict(list)
  add_data(pos_urls, pos_files, url_to_file)
  add_data(neg_urls, neg_files, url_to_file)
  urls = sorted(list(url_to_file))
  random.shuffle(urls)
  total_files = sum(len(x) for x in url_to_file.values())
  print('Found %d urls, %d files' % (len(urls), total_files))
  num_train = int(OPTS.train_frac * total_files)
  train_files = []
  dev_files = []
  for url in urls:
    files = url_to_file[url]
    if len(train_files) + len(files) <= num_train:
      train_files.extend(url_to_file[url])
    else:
      dev_files.extend(url_to_file[url])
  write_data('train', train_files)
  write_data('dev', dev_files)

if __name__ == '__main__':
  OPTS = parse_args()
  main()

