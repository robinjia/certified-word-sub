"""Analyze word vectors."""
import argparse
import os
from scipy.stats import special_ortho_group
import sys
import torch
from tqdm import tqdm

sys.path.append(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import data_util
import text_classification
import vocabulary

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('model_dir')
  parser.add_argument('checkpoint', type=int)
  parser.add_argument('--data_cache_dir')
  parser.add_argument('--epsilon', '-e', type=float, default=1.0)
  parser.add_argument('--dimension', '-d', type=int, default=100)
  parser.add_argument('--num-matrices', '-T', type=int, default=20)
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def log_box_volume(vecs):
  lb, _ = torch.min(vecs, dim=0)
  ub, _ = torch.max(vecs, dim=0)
  return torch.sum(torch.log(ub - lb))

def find_within_box(mat, indices, vocab, epsilon=1.0):
  cur_vecs = torch.stack([mat[i,:] for i in indices])  # n, d
  lb = torch.min(cur_vecs, dim=0)[0]
  ub = torch.max(cur_vecs, dim=0)[0]
  within_box = (torch.min(lb <= mat, dim=1)[0] & torch.min(mat <= ub, dim=1)[0]).nonzero()
  return sorted([vocab.get_word(i.item()) for i in within_box])

def measure_size(mat, indices, stdevs):
  cur_vecs = torch.stack([mat[i,:] for i in indices])  # n, d
  lb = torch.min(cur_vecs, dim=0)[0]
  ub = torch.max(cur_vecs, dim=0)[0]
  num_stdevs = (ub - lb) / (stdevs + 1e-7)
  return torch.mean(num_stdevs)

def main():
  OPTS.use_toy_data = False
  OPTS.use_lm = False
  OPTS.neighbor_file = data_util.NEIGHBOR_FILE
  OPTS.imdb_dir = text_classification.IMDB_DIR
  OPTS.test = False
  OPTS.glove_dir = vocabulary.GLOVE_DIR
  OPTS.glove = '840B.300d'
  OPTS.downsample_to = None
  OPTS.downsample_shard = 0
  OPTS.truncate_to = None
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  print('Loading model', file=sys.stderr)
  model_state = dict(torch.load(os.path.join(OPTS.model_dir, 'model-checkpoint-%d.pth' % OPTS.checkpoint)))
  input_layer = model_state['linear_input.weight']
  print('Loading data', file=sys.stderr)
  train_data, dev_data, glove_mat, attack_surface = text_classification.load_datasets(device, OPTS)
  learned_mat = torch.matmul(glove_mat, torch.t(input_layer))
  vocab = train_data.vocab
  neighbors = attack_surface.neighbors
  words = vocab.word_list
  glove_scale = torch.std(glove_mat, dim=0)
  learned_scale = torch.std(learned_mat, dim=0)
  learned_smaller = 0
  total = 0
  for w in tqdm(words):
    if w not in neighbors: continue
    if not neighbors[w]: continue
    print('Word: %s' % w)
    cur_words = [w] + neighbors[w]
    indices = [vocab.get_index(x) for x in cur_words]
    glove_contained = find_within_box(glove_mat, indices, vocab, OPTS.epsilon)
    learned_contained = find_within_box(learned_mat, indices, vocab, OPTS.epsilon)
    print('  Neighborhood is  %d words: [%s]' % (len(cur_words), ', '.join(sorted(cur_words))))
    print('  GloVe contains   %d words: [%s]' % (len(glove_contained), ', '.join(glove_contained)))
    print('  Learned contains %d words: [%s]' % (len(learned_contained), ', '.join(learned_contained)))
    glove_size = measure_size(glove_mat, indices, glove_scale)
    learned_size = measure_size(learned_mat, indices, learned_scale)
    print('  GloVe   size: %.2f' % glove_size)
    print('  Learned size: %.2f' % learned_size)
    print()
    if learned_size < glove_size:
      learned_smaller += 1
    total += 1
    print('  So far: Learned smaller on %d/%d = %.2f%% words' % (
        learned_smaller, total, 100 * learned_smaller / total))


if __name__ == '__main__':
  OPTS = parse_args()
  main()

