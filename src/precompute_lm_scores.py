"""Precompute language model scores on dev data."""
import argparse
import json
import os
import sys
import torch
from tqdm import tqdm

import data_util
import entailment
import text_classification
from train import TASK_CLASSES
import vocabulary

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                'windweller-lw2/adaptive_softmax'))
import query as lmquery

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser('Precompute language model scores.')
  parser.add_argument('task', choices=TASK_CLASSES.keys())
  parser.add_argument('split', choices=['train', 'dev', 'test'])
  parser.add_argument('out_file')
  parser.add_argument('--num-examples', '-n', type=int)
  parser.add_argument('--shard', '-s', type=int, help='Shard index', default=None)
  parser.add_argument('--shard-size', type=int, default=5000)
  parser.add_argument('--window-radius', '-w', type=int, default=6)
  parser.add_argument('--neighbor-file', type=str, default=data_util.NEIGHBOR_FILE)
  parser.add_argument('--imdb-dir', type=str, default=text_classification.IMDB_DIR)
  parser.add_argument('--snli-dir', type=str, default=entailment.SNLI_DIR)
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def main():
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  query_handler = lmquery.load_model(device)
  with open(OPTS.neighbor_file) as f:
    neighbors = json.load(f)
  if OPTS.task == 'classification':
    raw_data = text_classification.IMDBDataset.get_raw_data(
        OPTS.imdb_dir, test=(OPTS.split == 'test'))
  elif OPTS.task == 'entailment':
    OPTS.test = (OPTS.split == 'test')
    OPTS.adv_only = False
    raw_data = entailment.SNLIDataset.get_raw_data(OPTS)
  else:
    raise NotImplementedError
  if OPTS.split == 'train':
    data = raw_data.train_data
  else:  # dev or test
    data = raw_data.dev_data
  if OPTS.num_examples:
    data = data[:OPTS.num_examples]
  if OPTS.shard is not None:
    print('Restricting to shard %d' % OPTS.shard)
    data = data[OPTS.shard * OPTS.shard_size:(OPTS.shard + 1) * OPTS.shard_size]
  with open(OPTS.out_file, 'w') as f:
    for sent_idx, example in enumerate(tqdm(data)):
      if OPTS.task == 'classification':
        sentence = example[0]
      elif OPTS.task == 'entailment':
        sentence = example[0][1]  # Only look at hypothesis
      print('%d\t%s' % (sent_idx, sentence), file=f)
      words = sentence.split(' ')
      for i, w in enumerate(words):
        if w in neighbors:
          options = [w] + neighbors[w]
          start = max(0, i - OPTS.window_radius)
          end = min(len(words), i + 1 + OPTS.window_radius)
          # Remove OOV words from prefix and suffix
          prefix = [x for x in words[start:i] if x in query_handler.word_to_idx]
          suffix = [x for x in words[i+1:end] if x in query_handler.word_to_idx]
          queries = []
          in_vocab_options = []
          for opt in options:
            if opt in query_handler.word_to_idx:
              queries.append(prefix + [opt] + suffix)
              in_vocab_options.append(opt)
            else:
              print('%d\t%d\t%s\t%s' % (sent_idx, i, opt, float('-inf')), file=f)
          if queries:
            log_probs = query_handler.query(queries, batch_size=16)
            for x, lp in zip(in_vocab_options, log_probs):
              print('%d\t%d\t%s\t%s' % (sent_idx, i, x, lp), file=f)
      f.flush()

if __name__ == '__main__':
  OPTS = parse_args()
  main()

