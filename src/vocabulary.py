"""A basic vocabulary class."""
import collections
import os
import numpy as np
import torch
from tqdm import tqdm

UNK_TOKEN = '<UNK>'
UNK_INDEX = 0

NULL_TOKEN = '<NULL>'
NULL_INDEX = 1

GLOVE_DIR = 'data/glove'
GLOVE_CONFIGS = {
    '6B.50d': {'size': 50, 'lines': 400000},
    '840B.300d': {'size': 300, 'lines': 2196017}
}


class Vocabulary(object):
  @classmethod
  def read_word_vecs(cls, word_set, glove_dir, glove_name, device, normalize=False, prepend_null=False):
    vocab = cls(prepend_null=prepend_null)
    glove_config = GLOVE_CONFIGS[glove_name]
    vecs = [np.zeros(glove_config['size'])]  # UNK embedding, won't be used
    if prepend_null:
        vecs.append(np.zeros((300)))  # NULL embedding
    found = 0
    fn = os.path.join(glove_dir, 'glove.%s.txt' % glove_name)
    print('Reading GloVe vectors from %s...' % fn)
    with open(fn) as f:
      for i, line in tqdm(enumerate(f), total=glove_config['lines']):
        toks = line.strip().split(' ')
        word = toks[0]
        if word in word_set and word not in vocab:
          found += 1
          vocab.add_word_hard(word)
          vecs.append(np.array([float(x) for x in toks[1:]]))
    print('Found %d/%d words in %s' % (found, len(word_set), fn))
    word_mat = torch.tensor(vecs, dtype=torch.float, device=device)
    if normalize:
        word_mat = word_mat / word_mat.norm(dim=-1, keepdim=True)
    return vocab, word_mat

  def __init__(self, unk_threshold=0, prepend_null=False):
    """Initialize the vocabulary.

    Args:
      unk_threshold: words with <= this many counts will be considered <UNK>.
      prepend_null: if True index 1 will be <NULL>
    """
    self.unk_threshold = unk_threshold
    self.counts = collections.Counter()
    self.word2index = {UNK_TOKEN: UNK_INDEX}
    self.word_list = [UNK_TOKEN]
    if prepend_null:
        self.word2index[NULL_TOKEN] = NULL_INDEX
        self.word_list.append(NULL_TOKEN)

  def add_word(self, word, count=1):
    """Add a word (may still map to UNK if it doesn't pass unk_threshold)."""
    self.counts[word] += count
    if word not in self.word2index and self.counts[word] > self.unk_threshold:
      index = len(self.word_list)
      self.word2index[word] = index
      self.word_list.append(word)

  def add_words(self, words):
    for w in words:
      self.add_word(w)

  def add_sentence(self, sentence):
    self.add_words(sentence.split(' '))

  def add_sentences(self, sentences):
    for s in sentences:
      self.add_sentence(s)

  def add_word_hard(self, word):
    """Add word, make sure it is not UNK."""
    self.add_word(word, count=(self.unk_threshold+1))

  def get_word(self, index):
    return self.word_list[index]

  def get_index(self, word):
    if word in self.word2index:
      return self.word2index[word]
    return UNK_INDEX

  def indexify_sentence(self, sentence):
    return [self.get_index(w) for w in sentence.split(' ')]

  def indexify_list(self, elems):
    return [self.get_index(w) for w in elems]

  def recover_sentence(self, indices):
    return ' '.join(self.get_word(i) for i in indices)

  def has_word(self, word):
    return word in self.word2index

  def __contains__(self, word):
    return self.has_word(word)

  def size(self):
    # Report number of words that have been assigned an index
    return len(self.word2index)

  def __len__(self):
    return self.size()

  def __iter__(self):
    return iter(self.word_list)
