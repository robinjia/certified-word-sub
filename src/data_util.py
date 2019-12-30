"""Data handler classes and methods"""

import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler
import torch.nn.functional as F
import random


NEIGHBOR_FILE = 'data/counterfitted_neighbors.json'


def dict_batch_to_device(batch, device):
  """
  Moves a batch of data to device
  Args:
    - batch: Can be a Torch tensor or a dict where the values are torch tensors
    - device: A Torch device to move all the tensors to
  Returns:
    - a batch of the same type as input batch but on the device
      If a dict, also a dict with same keys
  """
  try:
    return batch.to(device)
  except AttributeError:
    # don't have a to function, must be a dict, recursively move to device
    return {k: dict_batch_to_device(v, device) for k, v in batch.items()}


class RawDataset(Dataset):
  """
  Dataset that only holds unprocessed text values
  Subsequent tasks should implement the get_word_set method for vocab picking
  """
  def __init__(self, train_data, dev_data):
    self.train_data = train_data
    self.dev_data = dev_data
    self.data = train_data + dev_data

  def __getitem__(self, index):
    return self.data[index]

  def __len__(self):
    return len(self.data)

  def get_word_set(self, neighbors):
    """
    Returns all the words found in this dataset
    and all their neighbors in a set
    """
    raise NotImplementedError


class ProcessedDataset(Dataset):
  """
  Dataset that holds processed examples
  Subsequent tasks should implement the methods defined below
  to interface with the rest of the training module
  """
  def __init__(self, raw_data, vocab, examples):
    self.raw_data = raw_data
    self.vocab = vocab
    self.examples = examples

  @classmethod
  def from_raw_data(self, raw_data, vocab, neighbors=None, truncate_to=None):
    """
    Given a RawDataset of examples, a vocab set and potentially dict of neigbors,
    initializes the dataset such that self.examples[i] corresponds to the ith example
    that can be passed to the relevant model
    """
    raise NotImplementedError

  @classmethod
  def get_raw_data(cls, *args, **kwargs):
    """
    Method that returns a RawClassificationDataset object
    that holds the entire dataset. This would later be passed to __init__
    """
    raise NotImplementedError

  @staticmethod
  def collate_examples(examples):
    """
    Method that takes a list of examples from self.examples and collates them into a single example
    with batched tensors
    """
    raise NotImplementedError

  @staticmethod
  def example_len(example):
    """
    Given an example returns the length of its principal sequence(s). Used for pooling samples of similar
    lengths into same batches to reduce the amount of padding necessary
    """
    raise NotImplementedError

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, index):
    return self.examples[index]

  def get_loader(self, batch_size):
    batch_sampler = PooledBatchSampler(self, batch_size, sort_key=self.example_len)
    return DataLoader(self, pin_memory=True, collate_fn=self.collate_examples, batch_sampler=batch_sampler)


def multi_dim_padded_cat(tensors, dim, padding_value=0):
  """
  Concatenates tensors along dim, padding elements to the largest length at
  each dimension. Assumes all tensors have the same dimensionality but makes no
  other assumptions about their size
  """
  if dim == 0:
     original_ordering = dim_first_ordering = list(range(len(tensors[0].shape)))
  else:
    # If dim is not 0, we make it so for ease later and re-permute at the end
    dims = list(range(len(tensors[0].shape)))
    dims.pop(dim)
    dim_first_ordering = [dim] + dims
    original_ordering = []
    for dim_idx in range(len(dim_first_ordering)):
      if dim_idx < dim:
        original_ordering.append(dim_idx + 1)
      elif dim_idx == dim:
        original_ordering.append(0)
      else:
        original_ordering.append(dim_idx)
  out_shape = []
  for in_dim in dim_first_ordering:
    out_shape.append(max(tensor.shape[in_dim] for tensor in tensors))
  out_shape[0] = sum(tensor.shape[dim] for tensor in tensors)
  out_tensor = tensors[0].new_empty(out_shape)
  cur_idx = 0
  for tensor in tensors:
    out_shape[0] = tensor.shape[dim]
    pad = []
    # see torch.nn.functional.pad documentation for why we need this format
    for tensor_dim, out_dim in list(zip(tensor.shape, out_shape))[:0:-1]:
      pad = pad + [0, out_dim - tensor_dim]
    out_tensor[cur_idx:cur_idx+out_shape[0], ...] = F.pad(tensor.permute(*dim_first_ordering), pad)
    cur_idx += out_shape[0]
  if dim != 0:
    out_tensor = out_tensor.permute(*original_ordering)
  return out_tensor


class PooledBatchSampler(BatchSampler):
  def __init__(self, dataset, batch_size, sort_within_batch=True, sort_key=len):
    self.dataset_lens = [sort_key(el) for el in dataset]
    self.batch_size = batch_size
    self.sort_within_batch = sort_within_batch

  def __iter__(self):
    """
    1- Partitions data indices into chunks of batch_size * 100
    2- Sorts each chunk by the sort_key
    3- Batches sorted chunks sequentially
    4- Shuffles the batches
    5- Yields each batch
    """
    idx_chunks = torch.split(torch.randperm(len(self.dataset_lens)), self.batch_size * 100)
    for idx_chunk in idx_chunks:
      sorted_chunk = torch.tensor(sorted(idx_chunk.tolist(), key=lambda idx: self.dataset_lens[idx]))
      chunk_batches = [chunk.tolist() for chunk in torch.split(sorted_chunk, self.batch_size)]
      random.shuffle(chunk_batches)
      for batch in chunk_batches:
        if self.sort_within_batch:
          batch.reverse()
        yield batch

  def __len__(self):
    return (len(self.dataset_lens) + self.batch_size - 1) // self.batch_size


class DataAugmenter(object):
  def __init__(self, augment_by):
    self.augment_by = augment_by

  def augment(self, dataset):
    raise NotImplementedError
