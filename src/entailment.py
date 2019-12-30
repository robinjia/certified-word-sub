"""IBP textual entailment model."""

from enum import Enum
import glob
import itertools
import json
import os
import pickle
import random

from nltk import word_tokenize
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import attacks
import data_util
import ibp
import vocabulary


SNLI_DIR = 'data/snli'
LM_FILE = 'data/lm_scores/snli_all.txt'
LOSS_FUNC = nn.BCEWithLogitsLoss()


class AdversarialModel(nn.Module):
  def __init__(self):
    super(AdversarialModel, self).__init__()

  def query(self, dataset, device, batch_size=1, return_bounds=False):
    """Query the model on a Dataset.

    Args:
      dataset: a Dataset.
      device: torch device.
      neighbors: if provided, pass this to Dataset().
      batch_size: batch size (default=1).

    Returns: Tensor of logits & gold labels
    """
    data = dataset.get_loader(batch_size)
    output = []
    gold = []
    with torch.no_grad():
      for batch in data:
        batch = data_util.dict_batch_to_device(batch, device)
        output.append(self.forward(batch, compute_bounds=return_bounds))
        gold.append(batch['y'])
    return ibp.cat(output, dim=0), ibp.cat(gold, dim=0)


class EntailmentLabels(Enum):
  contradiction = 0
  neutral = 1
  entailment = 2


class BOWModel(AdversarialModel):
  """Bag of words + MLP"""

  def __init__(self, word_vec_size, hidden_size, word_mat,
               dropout_prob=0.1, num_layers=3, no_wordvec_layer=False):
    super(BOWModel, self).__init__()
    self.embs = ibp.Embedding.from_pretrained(word_mat)
    self.rotation = ibp.Linear(word_vec_size, hidden_size)
    self.sum_drop = ibp.Dropout(dropout_prob) if dropout_prob else None
    layers = []
    for i in range(num_layers):
      layers.append(ibp.Linear(2*hidden_size, 2*hidden_size))
      layers.append(ibp.Activation(F.relu))
      if dropout_prob:
        layers.append(ibp.Dropout(dropout_prob))
    layers.append(ibp.Linear(2*hidden_size, len(EntailmentLabels)))
    layers.append(ibp.LogSoftmax(dim=1))
    self.layers = nn.Sequential(*layers)

  def forward(self, batch, compute_bounds=True, cert_eps=1.0):
    """
    Forward pass of BOWModel.
    Args:
      batch: A batch dict from an EntailmentDataset with the following keys:
        - prem: tensor of word vector indices for premise (B, p, 1)
        - hypo: tensor of word vector indices for hypothesis (B, h, 1)
        - prem_mask: binary mask over premise words (1 for real, 0 for pad), size (B, p)
        - hypo_mask: binary mask over hypothesis words (1 for real, 0 for pad), size (B, h)
        - prem_lengths: lengths of premises, size (B,)
        - hypo_lengths: lengths of hypotheses, size (B,)
      compute_bounds: If True compute the interval bounds and reutrn an IntervalBoundedTensor as logits. Otherwise just use the values
      cert_eps: float, scaling factor for the interval bounds.
    """
    def encode(sequence, mask):
      vecs = self.embs(sequence)
      vecs = self.rotation(vecs)
      if isinstance(vecs, ibp.DiscreteChoiceTensor):
        vecs = vecs.to_interval_bounded(eps=cert_eps)
      z1 = ibp.activation(F.relu, vecs)
      z1_masked = z1 * mask.unsqueeze(-1)
      z1_pooled = ibp.sum(z1_masked, -2)
      return z1_pooled
    if not compute_bounds:
        batch['prem']['x'] = batch['prem']['x'].val
        batch['hypo']['x'] = batch['hypo']['x'].val
    prem_encoded = encode(batch['prem']['x'], batch['prem']['mask'])
    hypo_encoded = encode(batch['hypo']['x'], batch['hypo']['mask'])
    input_encoded = ibp.cat([prem_encoded, hypo_encoded], -1)
    logits = self.layers(input_encoded)
    return logits


class DecompAttentionModel(AdversarialModel):
  """Decomposable Attention model from Parikh et al"""

  def __init__(self, word_vec_size, hidden_size, word_mat,
               dropout_prob=0.1, num_layers=2, no_wordvec_layer=False):
    super(DecompAttentionModel, self).__init__()
    self.embs = ibp.Embedding.from_pretrained(word_mat)
    self.null = nn.Parameter(torch.normal(mean=torch.zeros(word_vec_size)))
    self.rotation = None
    hidden_size = word_vec_size
    self.rotation = ibp.Linear(word_vec_size, hidden_size)

    def get_feedforward_layers(num_layers, input_size, hidden_size, output_size):
      layers = []
      for i in range(num_layers):
        layer_in_size = input_size if i == 0 else hidden_size
        layer_out_size = output_size if i == num_layers - 1 else hidden_size
        if dropout_prob:
          layers.append(ibp.Dropout(dropout_prob))
        layers.append(ibp.Linear(layer_in_size, layer_out_size))
        if i < num_layers - 1:
          layers.append(ibp.Activation(F.relu))
      return layers

    ff_layers = get_feedforward_layers(num_layers, hidden_size, hidden_size, 1)
    self.feedforward = nn.Sequential(*ff_layers)

    compare_layers = get_feedforward_layers(num_layers, 2 * hidden_size, hidden_size, hidden_size)
    self.compare_ff = nn.Sequential(*compare_layers)

    output_layers = get_feedforward_layers(num_layers, 2 * hidden_size, hidden_size, hidden_size)
    output_layers.append(ibp.Linear(hidden_size, len(EntailmentLabels)))
    output_layers.append(ibp.LogSoftmax(dim=1))
    self.output_layer = nn.Sequential(*output_layers)

  def forward(self, batch, compute_bounds=True, cert_eps=1.0):
    """
    Forward pass of DecompAttentionModel.
    Args:
      batch: A batch dict from an EntailmentDataset with the following keys:
        - prem: tensor of word vector indices for premise (B, p, 1)
        - hypo: tensor of word vector indices for hypothesis (B, h, 1)
        - prem_mask: binary mask over premise words (1 for real, 0 for pad), size (B, p)
        - hypo_mask: binary mask over hypothesis words (1 for real, 0 for pad), size (B, h)
        - prem_lengths: lengths of premises, size (B,)
        - hypo_lengths: lengths of hypotheses, size (B,)
      compute_bounds: If True compute the interval bounds and reutrn an IntervalBoundedTensor as logits. Otherwise just use the values
      cert_eps: float, scaling factor for the interval bounds.
    """
    def encode(sequence, mask):
      vecs = self.embs(sequence)
      if isinstance(vecs, ibp.DiscreteChoiceTensor):
        null = torch.zeros_like(vecs.val[0])
        null_choice = torch.zeros_like(vecs.choice_mat[0])
        null[0] = self.null
        null_choice[0, 0] = self.null
        vecs.val = vecs.val + null
        vecs.choice_mat = vecs.choice_mat + null_choice
      else:
        null = torch.zeros_like(vecs[0])
        null[0] = self.null
        vecs = vecs + null
      vecs = self.rotation(vecs)
      if isinstance(vecs, ibp.DiscreteChoiceTensor):
        vecs = vecs.to_interval_bounded(eps=cert_eps)
      return ibp.activation(F.relu, vecs) * mask.unsqueeze(-1)

    if not compute_bounds:
        batch['prem']['x'] = batch['prem']['x'].val
        batch['hypo']['x'] = batch['hypo']['x'].val
    prem_encoded = encode(batch['prem']['x'], batch['prem']['mask']) # (bXpXe)
    hypo_encoded = encode(batch['hypo']['x'], batch['hypo']['mask']) # (bXhXe)
    prem_weights = self.feedforward(prem_encoded) * batch['prem']['mask'].unsqueeze(-1) # (bXpX1)
    hypo_weights = self.feedforward(hypo_encoded) * batch['hypo']['mask'].unsqueeze(-1) # (bXhX1)
    attention = ibp.bmm(prem_weights, hypo_weights.permute(0,2,1)) # (bXpX1) X (bX1Xh) => (bXpXh)
    attention_mask = batch['prem']['mask'].unsqueeze(-1) * batch['hypo']['mask'].unsqueeze(1)
    attention_masked = ibp.add(attention, (1 - attention_mask) * -1e20)
    attended_prem = self.attend_on(hypo_encoded, prem_encoded, attention_masked) # (bXpX2e)
    attended_hypo = self.attend_on(prem_encoded, hypo_encoded, attention_masked.permute(0,2,1)) # (bXhX2e)
    compared_prem = self.compare_ff(attended_prem) * batch['prem']['mask'].unsqueeze(-1) # (bXpXhid)
    compared_hypo = self.compare_ff(attended_hypo) * batch['hypo']['mask'].unsqueeze(-1) # (bXhXhid)
    prem_aggregate = ibp.pool(torch.sum, compared_prem, dim=1) # (bXhid)
    hypo_aggregate = ibp.pool(torch.sum, compared_hypo, dim=1) # (bXhid)
    aggregate = ibp.cat([prem_aggregate, hypo_aggregate], dim=-1) # (bX2hid)
    return self.output_layer(aggregate) # (b)

  def attend_on(self, source, target, attention):
    """
    Args:
      - source: (bXsXe)
      - target: (bXtXe)
      - attention: (bXtXs)
    """
    attention_logsoftmax = ibp.log_softmax(attention, 1)
    attention_normalized = ibp.activation(torch.exp, attention_logsoftmax)
    attended_target = ibp.matmul_nneg(attention_normalized, source) # (bXtXe)
    return ibp.cat([target, attended_target], dim=-1)

def load_model(word_mat, device, opts):
  """
  Try to load a model on the device given the word_mat and opts.
  Tries to load a model from the given or latest checkpoint if specified in the opts.
  Otherwise instantiates a new model on the device.
  """
  vec_size = vocabulary.GLOVE_CONFIGS[opts.glove]['size']
  if opts.model == 'bow':
    model = BOWModel(
        vec_size, vec_size, word_mat, num_layers=opts.num_layers, dropout_prob=opts.dropout_prob, no_wordvec_layer=opts.no_wordvec_layer).to(device)
  if opts.model == 'decomp-attn':
    model = DecompAttentionModel(
        vec_size, opts.hidden_size, word_mat, dropout_prob=opts.dropout_prob, num_layers=opts.num_layers, no_wordvec_layer=opts.no_wordvec_layer).to(device)
  if opts.load_dir:
    try:
      if opts.load_ckpt is None:
        load_fn = sorted(glob.glob(os.path.join(opts.load_dir, 'model-checkpoint-[0-9]+.pth')))[-1]
      else:
        load_fn = os.path.join(opts.load_dir, 'model-checkpoint-%d.pth' % opts.load_ckpt)
      print('Loading model from %s.' % load_fn)
      # Cache the word vectors before loading to avoid size mismatches
      state_dict = dict(torch.load(load_fn))
      if opts.prepend_null:
          null_vec = state_dict['embs.weight'][vocabulary.NULL_INDEX]
          model.embs.weight[vocabulary.NULL_INDEX] = null_vec
      state_dict['embs.weight'] = model.embs.weight
      model.load_state_dict(state_dict, strict=False)
      print('Finished loading model.')
    except Exception as ex:
      print("Couldn't load model, starting anew: {}".format(ex))
  return model


def load_datasets(device, opts):
  """
  Loads entailment datasets given opts on the device and returns the dataset.
  If a data cache is specified in opts and the cached data there is of the same class
    as the one specified in opts, uses the cache. Otherwise reads from the raw dataset
    files specified in OPTS.
  Returns:
    - train_data:  EntailmentDataset - Processed training dataset
    - dev_data: Optional[EntailmentDataset] - Processed dev dataset if raw dev data was found or
        dev_frac was specified in opts
    - word_mat: torch.Tensor
    - attack_surface: AttackSurface - defines the adversarial attack surface
  """
  data_class = ToyEntailmentDataset if opts.use_toy_data else SNLIDataset
  try:
    if opts.adv_only:
      train_data = None
      train_attack_surface = None
    else:
      with open(os.path.join(opts.data_cache_dir, 'train_data.pkl'), 'rb') as infile:
        train_data = pickle.load(infile)
        if not isinstance(train_data, data_class):
          raise Exception("Cached dataset of wrong class: {}".format(type(train_data)))
      with open(os.path.join(opts.data_cache_dir, 'train_attack_surface.pkl'), 'rb') as infile:
        train_attack_surface = pickle.load(infile)
    with open(os.path.join(opts.data_cache_dir, 'dev_data.pkl'), 'rb') as infile:
      dev_data = pickle.load(infile)
      if not isinstance(dev_data, data_class):
        raise Exception("Cached dataset of wrong class: {}".format(type(train_data)))
    with open(os.path.join(opts.data_cache_dir, 'word_mat.pkl'), 'rb') as infile:
      word_mat = pickle.load(infile)
    with open(os.path.join(opts.data_cache_dir, 'dev_attack_surface.pkl'), 'rb') as infile:
      dev_attack_surface = pickle.load(infile)
    print("Loaded data from {}.".format(opts.data_cache_dir))
  except Exception as ex:
    print('Couldn\'t load data from cache: {}, reading from raw files'.format(ex))
    if opts.adv_only:
      train_data = None
      train_attack_surface = None
    else:
      train_attack_surface = attacks.WordSubstitutionAttackSurface.from_file(opts.neighbor_file)
    if opts.use_lm:
      dev_attack_surface = attacks.LMConstrainedAttackSurface.from_files(
          opts.neighbor_file, opts.snli_lm_file)
    else:
      dev_attack_surface = attacks.WordSubstitutionAttackSurface.from_file(opts.neighbor_file)
    raw_data = data_class.get_raw_data(opts)
    word_set = raw_data.get_word_set(train_attack_surface, dev_attack_surface=dev_attack_surface)
    vocab, word_mat = vocabulary.Vocabulary.read_word_vecs(word_set, opts.glove_dir, opts.glove, device, prepend_null=opts.prepend_null, normalize=opts.normalize_word_vecs)
    if not opts.adv_only:
      train_data = data_class.from_raw_data(raw_data.train_data, vocab, attack_surface=train_attack_surface,
                                            downsample_to=opts.downsample_to, prepend_null=opts.prepend_null, use_tqdm=True, downsample_shard=opts.downsample_shard)
    dev_data = data_class.from_raw_data(raw_data.dev_data, vocab, attack_surface=dev_attack_surface,
                                        downsample_to=opts.downsample_to, prepend_null=opts.prepend_null, use_tqdm=True, downsample_shard=opts.downsample_shard)
    if opts.data_cache_dir:
      if not opts.adv_only:
        with open(os.path.join(opts.data_cache_dir, 'train_data.pkl'), 'wb') as outfile:
          pickle.dump(train_data, outfile)
        with open(os.path.join(opts.data_cache_dir, 'train_attack_surface.pkl'), 'wb') as outfile:
          pickle.dump(train_attack_surface, outfile)
      with open(os.path.join(opts.data_cache_dir, 'dev_data.pkl'), 'wb') as outfile:
        pickle.dump(dev_data, outfile)
      with open(os.path.join(opts.data_cache_dir, 'word_mat.pkl'), 'wb') as outfile:
        pickle.dump(word_mat, outfile)
      with open(os.path.join(opts.data_cache_dir, 'dev_attack_surface.pkl'), 'wb') as outfile:
        pickle.dump(dev_attack_surface, outfile)
  return train_data, dev_data, word_mat, dev_attack_surface


def get_margins(model_output, gold_labels):
  if isinstance(model_output, ibp.IntervalBoundedTensor):
    logits = model_output.val
    w_true_class_pred = (model_output.lb * gold_labels).sum(dim=1)
    w_highest_false_pred = (model_output.ub + (gold_labels * -1e20)).max(dim=1)[0]
    w_value_margin = w_true_class_pred - w_highest_false_pred
  else:
    logits = model_output
    w_value_margin = None
  true_class_pred = (logits * gold_labels).sum(dim=1)
  highest_false_pred = (logits + (gold_labels * -1e20)).max(dim=1)[0]
  value_margin = true_class_pred - highest_false_pred
  return value_margin, w_value_margin


def compute_is_correct(model_output, gold_labels):
  if isinstance(model_output, ibp.IntervalBoundedTensor):
    logits = model_output.val
    # Worst case pred. is the LB of correct class
    # combined with the UBs of the other classes
    worst_case_pred = (gold_labels * model_output.lb + (1 - gold_labels) * model_output.ub).argmax(dim=1)
    gold_labels = gold_labels.argmax(dim=1)
    cert_correct = ((worst_case_pred - gold_labels) == 0)
  else:
    gold_labels = gold_labels.argmax(dim=1)
    logits = model_output
    cert_correct = None
  predictions = logits.argmax(dim=1)
  correct = ((predictions - gold_labels) == 0)
  return correct, cert_correct


def num_correct(model_output, gold_labels):
  """
  Given the output of model and gold labels returns number of correct and certified correct
  predictions
  Args:
    - model_output: output of the model, could be ibp.IntervalBoundedTensor or torch.Tensor
    - gold_labels: torch.Tensor
  Returns:
    - num_correct: int - number of correct predictions from the actual model output
    - num_cert_correct - number of bounds-certified correct predictions if the model_output was an
        IntervalBoundedTensor, 0 otherwise.
  """
  is_correct, is_cert_correct = compute_is_correct(model_output, gold_labels)
  num_correct = is_correct.sum().item()
  if is_cert_correct is not None:
    num_cert_correct = is_cert_correct.sum().item()
  else:
    num_cert_correct = 0
  return num_correct, num_cert_correct


class RawEntailmentDataset(data_util.RawDataset):
  """
  Dataset that only holds (prem, hypo) ,y as ((str, str), str) tuples
  """
  def get_word_set(self, train_attack_surface, dev_attack_surface=None):
    if dev_attack_surface is None:
      dev_attack_surface = train_attack_surface
    word_set = set()
    for x, y in self.train_data:
      prem, hypo = x
      prem_words = prem.split()
      hypo_words = hypo.split()
      words = prem_words + hypo_words
      for w in words:
        word_set.add(w)
      try:
        swaps = train_attack_surface.get_swaps(hypo_words)
        for cur_swaps in swaps:
          for w in cur_swaps:
            word_set.add(w)
      except KeyError:
        pass
    for x, y in self.dev_data:
      prem, hypo = x
      prem_words = prem.split()
      hypo_words = hypo.split()
      words = prem_words + hypo_words
      for w in words:
        word_set.add(w)
      try:
        swaps = dev_attack_surface.get_swaps(hypo_words)
        for cur_swaps in swaps:
          for w in cur_swaps:
            word_set.add(w)
      except KeyError:
        pass
    return word_set


class EntailmentDataset(data_util.ProcessedDataset):
  """
  Dataset that holds processed example dicts
  """
  @classmethod
  def from_raw_data(cls, raw_data, vocab, attack_surface=None, truncate_to=None, downsample_to=None, prepend_null=False, use_tqdm=False, downsample_shard=None):
    if truncate_to:
      raise NotImplementedError  # Probably never needed since SNLI sentences are so short
    if downsample_to:
      if downsample_shard is None:
        downsample_shard = 0
      raw_data = raw_data[downsample_shard * downsample_to:(downsample_shard + 1) * downsample_to]
    examples = []
    iteration = tqdm(raw_data) if use_tqdm else raw_data
    for inpt, y in iteration:
      try:
        examples.append(cls.process_example(inpt, y, vocab, attack_surface, prepend_null=prepend_null))
      except ValueError as err:
        print('Error processing example: {} Skipping.'.format(err))
    return cls(raw_data, vocab, examples)

  @classmethod
  def process_example(cls, inpt, y, vocab, attack_surface, skip_prem=True, prepend_null=False):
    example = {}
    for idx,sequence in enumerate(['prem', 'hypo']):
      x = inpt[idx]
      all_words = x.split()
      if prepend_null:
          all_words = ['<NULL>'] + all_words
      words = [w for w in all_words if w in vocab]  # Delete UNK words
      word_idxs = [vocab.get_index(w) for w in words]
      if len(word_idxs) < 1:
        raise ValueError("Sequence:\n\t{}\n is all UNK words in sample:\n \t{}\n".format(x, inpt))
      x_torch = torch.tensor(word_idxs).view(1, -1, 1) # (1, T, d)
      if attack_surface and not (skip_prem and sequence == 'prem'):
        swap_words = all_words[1:] if prepend_null else all_words  # Don't try to swap NULL
        all_swaps = attack_surface.get_swaps(swap_words)
        if prepend_null:
          all_swaps = [[]] + all_swaps  # Add an empty swaps list at index 0 for NULL
        swaps = [s for w, s in zip(all_words, all_swaps) if w in vocab]
        choices = [[w] + cur_swaps for w, cur_swaps in zip(words, swaps)]
        choices_word_idxs = [
          torch.tensor([vocab.get_index(c) for c in c_list], dtype=torch.long) for c_list in choices
        ]
        if any(0 in choices.view(-1).tolist() for choices in choices_word_idxs):
          raise ValueError("UNK tokens found")
        choices_torch = pad_sequence(choices_word_idxs, batch_first=True).unsqueeze(2).unsqueeze(0) # (1, T, C, 1)
        choices_mask = (choices_torch.squeeze(-1) != 0).long()  # (1, T, C)
      else:
        choices_torch = x_torch.view(1, -1, 1, 1) # (1, T, 1, 1)
        choices_mask = torch.ones_like(x_torch.view(1, -1, 1))
      mask_torch = torch.ones((1, len(word_idxs)))
      x_bounded = ibp.DiscreteChoiceTensor(x_torch, choices_torch, choices_mask, mask_torch)
      lengths_torch = torch.tensor(len(word_idxs)).view(1)
      example[sequence] = dict(x=x_bounded, mask=mask_torch, lengths=lengths_torch)
    example['y'] = torch.zeros((1, len(EntailmentLabels)), dtype=torch.float)
    example['y'][0, y.value] = 1
    return example

  @staticmethod
  def example_len(example):
    """
    Should sort by first hypothesis length then premise length
    """
    return (example['hypo']['x'].shape[1], example['prem']['x'].shape[1])

  @staticmethod
  def collate_examples(examples):
    """
    Turns a list of examples into a workable batch:
    """
    if len(examples) == 1:
      return examples[0]
    B = len(examples)

    max_prem_len = max(ex['prem']['x'].shape[1] for ex in examples)
    prem_vals = []
    prem_choice_mats = []
    prem_choice_masks = []
    prem_lengths = torch.zeros((B, ), dtype=torch.long)
    prem_masks = torch.zeros((B, max_prem_len))

    max_hypo_len = max(ex['hypo']['x'].shape[1] for ex in examples)
    hypo_vals = []
    hypo_choice_mats = []
    hypo_choice_masks = []
    hypo_lengths = torch.zeros((B, ), dtype=torch.long)
    hypo_masks = torch.zeros((B, max_hypo_len))

    gold_ys = []
    for i, ex in enumerate(examples):
      prem_vals.append(ex['prem']['x'].val)
      prem_choice_mats.append(ex['prem']['x'].choice_mat)
      prem_choice_masks.append(ex['prem']['x'].choice_mask)
      cur_prem_len = ex['prem']['x'].shape[1]
      prem_masks[i, :cur_prem_len] = 1
      prem_lengths[i] = ex['prem']['lengths'][0]

      hypo_vals.append(ex['hypo']['x'].val)
      hypo_choice_mats.append(ex['hypo']['x'].choice_mat)
      hypo_choice_masks.append(ex['hypo']['x'].choice_mask)
      cur_hypo_len = ex['hypo']['x'].shape[1]
      hypo_masks[i, :cur_hypo_len] = 1
      hypo_lengths[i] = ex['hypo']['lengths'][0]

      gold_ys.append(ex['y'])
    prem_vals = data_util.multi_dim_padded_cat(prem_vals, 0).long()
    prem_choice_mats = data_util.multi_dim_padded_cat(prem_choice_mats, 0).long()
    prem_choice_masks = data_util.multi_dim_padded_cat(prem_choice_masks, 0).long()

    hypo_vals = data_util.multi_dim_padded_cat(hypo_vals, 0).long()
    hypo_choice_mats = data_util.multi_dim_padded_cat(hypo_choice_mats, 0).long()
    hypo_choice_masks = data_util.multi_dim_padded_cat(hypo_choice_masks, 0).long()

    y = torch.cat(gold_ys, 0)
    return {
        'prem': {
          'x': ibp.DiscreteChoiceTensor(prem_vals, prem_choice_mats, prem_choice_masks, prem_masks),
          'mask': prem_masks, 'lengths': prem_lengths},
        'hypo': {
          'x': ibp.DiscreteChoiceTensor(hypo_vals, hypo_choice_mats, hypo_choice_masks, hypo_masks),
          'mask': hypo_masks, 'lengths': hypo_lengths},
        'y': y}


class ToyEntailmentDataset(EntailmentDataset):
  """
  Dataset that holds toy entailment data
  """
  @classmethod
  def get_raw_data(cls, *args, **kwargs):
    data = [
        (('man running', 'human moving'), EntailmentLabels.entailment),
        (('man running', 'man running'), EntailmentLabels.entailment),
        (('man running', 'human moving'), EntailmentLabels.entailment),
    ]
    return RawEntailmentDataset(data, data)


class SNLIDataset(EntailmentDataset):
  """
  Dataset that holds the SNLI sentiment classification data
  """
  @classmethod
  def get_raw_data(cls, opts):
    splits = {}
    # Number of examples in each split for better tqdm
    totals = {'train': 550152, 'dev': 10000, 'test': 10000}
    if opts.test:
      split_names = ['train', 'test']
    else:
      split_names = ['train', 'dev']
    for split in split_names:
      if opts.adv_only and split == 'train':
        splits['train'] = []
        continue
      data = []
      fn = os.path.join(opts.snli_dir, 'snli_1.0_{}.jsonl'.format(split))
      with open(fn) as f:
        for line in tqdm(f, total=totals[split]):
          example = json.loads(line)
          prem, hypo, gold_label = example['sentence1'], example['sentence2'], example['gold_label']
          prem_tokenized = ' '.join(word_tokenize(prem))
          hypo_tokenized = ' '.join(word_tokenize(hypo))
          try:
            gold_label = EntailmentLabels[gold_label]
          except KeyError:
            # Encountered gold label '-', can't use so skip it
            continue
          data.append(((prem_tokenized, hypo_tokenized), gold_label))
      random.shuffle(data)
      splits[split] = data
    return RawEntailmentDataset(*[splits[n] for n in split_names])


class DataAugmenter(data_util.DataAugmenter):
  def augment(self, dataset):
    new_examples = []
    for ex in tqdm(dataset.examples):
      new_examples.append(ex)
      x_orig = ex['hypo']['x']  # (1, T, 1)
      choices = []
      for i in range(x_orig.shape[1]):
        cur_choices = torch.masked_select(
            x_orig.choice_mat[0,i,:,0], x_orig.choice_mask[0,i,:].type(torch.uint8))
        choices.append(cur_choices)
      for t in range(self.augment_by):
        x_new = torch.stack([choices[i][random.choice(range(len(choices[i])))]
                             for i in range(len(choices))]).view(1, -1, 1)
        x_bounded = ibp.DiscreteChoiceTensor(
            x_new, x_orig.choice_mat, x_orig.choice_mask, x_orig.sequence_mask)
        ex_new = dict(ex)
        ex_new['hypo'] = dict(ex['hypo'])
        ex_new['hypo']['x'] = x_bounded
        new_examples.append(ex_new)
    return EntailmentDataset(None, dataset.vocab, new_examples)


class Adversary(object):
  """An Adversary tries to fool a model on a given example."""
  def __init__(self, attack_surface):
    self.attack_surface = attack_surface

  def run(self, model, dataset, device, opts=None):
    """Run adversary on a dataset.

    Args:
      model: a TextClassificationModel.
      dataset: a TextClassificationDataset.
      device: torch device.
    Returns: pair of
      - list of 0-1 adversarial loss of same length as |dataset|
      - list of list of adversarial examples (each is just a text string)
    """
    raise NotImplementedError


class ExhaustiveAdversary(Adversary):
  """An Adversary that exhaustively tries all allowed perturbations.

  Only practical for short sentences.
  """
  def run(self, model, dataset, device, opts=None):
    model.eval()
    is_correct = []
    adv_exs = []
    for x, y in dataset.raw_data:
      prem, hypo = x
      words = hypo.split()
      swaps = self.attack_surface.get_swaps(words)
      choices = [[w] + cur_swaps for w, cur_swaps in zip(words, swaps)]
      prod = 1
      for c in choices:
        prod *= len(c)
      print('ExhaustiveAdversary: "%s" -> %d options' % (hypo, prod))
      all_raw = [(prem, (' '.join(hypo_new)), y) for hypo_new in itertools.product(*choices)]
      cur_dataset = EntailmentDataset.from_raw_data(all_raw, dataset.vocab)
      preds, gold = model.query(cur_dataset, device)
      model_correct, model_cert_correct = compute_is_correct(preds, gold)
      cur_adv_exs = [all_raw[i][0] for i, p in enumerate(model_correct)
                     if p.item()]
      print(cur_adv_exs)
      adv_exs.append(cur_adv_exs)
      is_correct.append(int(len(cur_adv_exs) > 0))
    return is_correct, adv_exs


class GreedyAdversary(Adversary):
  """An adversary that picks a random word and greedily tries perturbations."""
  def __init__(self, attack_surface, num_epochs=10, num_tries=2, margin_goal=0.0):
    super(GreedyAdversary, self).__init__(attack_surface)
    self.num_epochs = num_epochs
    self.num_tries = num_tries
    self.margin_goal = margin_goal

  def run(self, model, dataset, device, opts=None):
    model.eval()
    is_correct = []
    adv_exs = []
    for x, y in tqdm(dataset.raw_data):
      prem, hypo = x
      # First query the example itself
      orig_pred, orig_gold = model.query(EntailmentDataset.from_raw_data(
          [(x, y)], dataset.vocab, attack_surface=self.attack_surface), device, return_bounds=True)
      model_correct, model_cert_correct = compute_is_correct(orig_pred, orig_gold)
      if model_correct.sum().item() == 0:
        print('ORIGINAL PREDICTION WAS WRONG')
        is_correct.append(0)
        adv_exs.append(x)
        continue

      # Now run adversarial search
      words = hypo.split()
      swaps = self.attack_surface.get_swaps(words)
      choices = [[w] + cur_swaps for w, cur_swaps in zip(words, swaps)]
      found = False
      for try_idx in range(self.num_tries):
        cur_words = list(words)
        for epoch in range(self.num_epochs):
          word_idxs = list(range(len(choices)))
          random.shuffle(word_idxs)
          for i in word_idxs:
            cur_raw = []
            for w_new in choices[i]:
              cur_raw.append(((prem, ' '.join(cur_words[:i] + [w_new] + cur_words[i+1:])), y))
            cur_dataset = EntailmentDataset.from_raw_data(cur_raw, dataset.vocab)
            preds, gold = model.query(cur_dataset, device)
            _, margins, _ = get_margins(preds, gold)
            best_idx = margins.argmin()
            best_idx = min(enumerate(margins), key=lambda x: x[1])[0]
            cur_words[i] = choices[i][best_idx]
            if margins[best_idx] < self.margin_goal:
              found = True
              is_correct.append(0)
              adv_exs.append([' '.join(cur_words)])
              print('ADVERSARY SUCCESS on ("%s", %s): Found "%s" with margin %.2f' % (x, y, adv_exs[-1], margins[best_idx]))
              if model_cert_correct.sum().item() > 0:
                print('^^ CERT CORRECT THOUGH')
              break
          if found: break
        if found: break
      else:
        is_correct.append(1)
        adv_exs.append([])
        print('ADVERSARY FAILURE on ("%s", %s)' % (x, y))
    return is_correct, adv_exs


class GeneticAdversary(Adversary):
  """An adversary that runs a genetic attack."""
  def __init__(self, attack_surface, num_iters=20, pop_size=60, margin_goal=0.0):
    super(GeneticAdversary, self).__init__(attack_surface)
    self.num_iters = num_iters
    self.pop_size = pop_size
    self.margin_goal = margin_goal

  def perturb(self, prem, hypo, choices, model, y, vocab, device, prepend_null=False):
    if all(len(c) == 1 for c in choices):
      value_margin, _ = get_margins(*(model.query(EntailmentDataset.from_raw_data([((prem, ' '.join(hypo)), y)], vocab, prepend_null=prepend_null), device)))
      return hypo, value_margin.item()
    good_idxs = [i for i, c in enumerate(choices) if len(c) > 1]
    idx = random.sample(good_idxs, 1)[0]
    best_replacement = None
    worst_margin = float('inf')
    for w_new in choices[idx]:
      cur_raw = [((prem, ' '.join(hypo[:idx] + [w_new] + hypo[idx+1:])), y)]
      cur_dataset = EntailmentDataset.from_raw_data(cur_raw, vocab, prepend_null=prepend_null)
      model_output, gold_labels = model.query(cur_dataset, device)
      value_margins, worst_case_margins = get_margins(model_output, gold_labels)
      if best_replacement is None or value_margins[0].item() < worst_margin:
          best_replacement = w_new
          worst_margin = value_margins[0].item()
    cur_words = list(hypo)
    cur_words[idx] = best_replacement
    return cur_words, worst_margin

  def run(self, model, dataset, device, opts=None):
    prepend_null = opts.prepend_null if opts is not None else False
    model.eval()
    is_correct = []
    adv_exs = []
    for x, y in tqdm(dataset.raw_data):
      # First query the example itself
      prem, hypo = x
      orig_pred, orig_gold = model.query(EntailmentDataset.from_raw_data(
          [(x, y)], dataset.vocab, attack_surface=self.attack_surface, prepend_null=prepend_null), device, return_bounds=True)
      model_correct, model_cert_correct = compute_is_correct(orig_pred, orig_gold)
      cert_correct = model_cert_correct.sum().item()
      value_margins, worst_case_margins = get_margins(orig_pred, orig_gold)
      print('Margin: %.6f, lower bound: %.6f, cert_correct=%s' % (
          value_margins[0].item(), worst_case_margins[0].item(), cert_correct))
      if model_correct.sum().item() <= 0:
        print('ORIGINAL PREDICTION WAS WRONG')
        is_correct.append(0)
        adv_exs.append(x)
        continue
      # Now run adversarial search
      hypo_words = hypo.split()
      swaps = self.attack_surface.get_swaps(hypo_words)
      choices = [[w] + cur_swaps for w, cur_swaps in zip(hypo_words, swaps)]
      found = False
      population = [self.perturb(prem, hypo_words, choices, model, y, dataset.vocab, device, prepend_null=prepend_null)
                    for i in range(self.pop_size)]
      for g in range(self.num_iters):
        best_idx = min(enumerate(population), key=lambda x: x[1][1])[0]
        print('Iteration %d: %.6f' % (g, population[best_idx][1]))
        if population[best_idx][1] < self.margin_goal:
          found = True
          is_correct.append(0)
          adv_exs.append(' '.join(population[best_idx][0]))
          print('ADVERSARY SUCCESS on ("%s", %s): Found "%s" with margin %.2f' % (x, y, adv_exs[-1], population[best_idx][1]))
          if cert_correct:
            print('^^ CERT CORRECT THOUGH')
          break
        new_population = [population[best_idx]]
        margins = np.array([m for c, m in population])
        adv_probs = 1 / (1 + np.exp(margins)) + 1e-6
            # Sigmoid of negative margin, for probabilty of wrong class
            # Add 1e-6 for numerical stability
        sample_probs = adv_probs / np.sum(adv_probs)
        for i in range(1, self.pop_size):
          parent1 = population[np.random.choice(range(len(population)), p=sample_probs)][0]
          parent2 = population[np.random.choice(range(len(population)), p=sample_probs)][0]
          child = [random.sample([w1, w2], 1)[0] for (w1, w2) in zip(parent1, parent2)]
          child_mut, new_margin = self.perturb(prem, child, choices, model, y,
                                               dataset.vocab, device, prepend_null=prepend_null)
          new_population.append((child_mut, new_margin))
        population = new_population
      else:
        is_correct.append(1)
        adv_exs.append([])
        print('ADVERSARY FAILURE on ("%s", %s)' % (x, y))
    return is_correct, adv_exs
