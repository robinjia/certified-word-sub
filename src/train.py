import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import torch

from tqdm import tqdm

import data_util
import entailment
import text_classification
import vocabulary


# Maps string keys to modules that hold the relevant functions for training against
# their tasks
TASK_CLASSES = {
  'classification': text_classification,
  'entailment': entailment
}


def train(task_class, model, train_data, num_epochs, lr, device, dev_data=None,
          cert_frac=0.0, initial_cert_frac=0.0, cert_eps=1.0, initial_cert_eps=0.0, non_cert_train_epochs=0, full_train_epochs=0,
          batch_size=1, epochs_per_save=1, augmenter=None, clip_grad_norm=0, weight_decay=0,
          save_best_only=False):
  print('Training model')
  sys.stdout.flush()
  loss_func = task_class.LOSS_FUNC
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
  zero_stats = {'epoch': 0, 'clean_acc': 0.0, 'cert_acc': 0.0}
  if augmenter:
    zero_stats['aug_acc'] = 0.0
  all_epoch_stats = {
    "loss": {"total": [],
             "clean": [],
             "cert": []},
    "cert": {"frac": [],
             "eps": []},
    "acc": {
      "train": {
        "clean": [],
        "cert": []},
      "dev": {
        "clean": [],
        "cert": []},
      "best_dev": {
        "clean": [zero_stats],
        "cert": [zero_stats]}},
    "total_epochs": num_epochs,
  }
  aug_dev_data = None
  if augmenter:
    all_epoch_stats['acc']['dev']['aug'] = []
    all_epoch_stats['acc']['best_dev']['aug'] = [zero_stats]
    print('Augmenting training data')
    aug_train_data = augmenter.augment(train_data)
    data = aug_train_data.get_loader(batch_size)
    if dev_data:
      print('Augmenting dev data')
      aug_dev_data = augmenter.augment(dev_data)  # Augment dev set now, for early stopping
  else:
    data = train_data.get_loader(batch_size)  # Create all batches now and pin them in memory
  # Linearly increase the weight of adversarial loss over all the epochs to end up at the final desired fraction
  cert_schedule = torch.tensor(np.linspace(initial_cert_frac, cert_frac, num_epochs - full_train_epochs - non_cert_train_epochs), dtype=torch.float, device=device)
  eps_schedule = torch.tensor(np.linspace(initial_cert_eps, cert_eps, num_epochs - full_train_epochs - non_cert_train_epochs), dtype=torch.float, device=device)
  for t in range(num_epochs):
    model.train()
    if t < non_cert_train_epochs:
        cur_cert_frac = 0.0
        cur_cert_eps = 0.0
    else:
        cur_cert_frac = cert_schedule[t - non_cert_train_epochs] if t - non_cert_train_epochs < len(cert_schedule) else cert_schedule[-1]
        cur_cert_eps = eps_schedule[t - non_cert_train_epochs] if t - non_cert_train_epochs < len(eps_schedule) else eps_schedule[-1]
    epoch = {
      "total_loss": 0.0,
      "clean_loss": 0.0,
      "cert_loss": 0.0,
      "num_correct": 0,
      "num_cert_correct": 0,
      "num": 0,
      "clean_acc": 0,
      "cert_acc": 0,
      "dev": {},
      "best_dev": {},
      "cert_frac": cur_cert_frac if isinstance(cur_cert_frac, float) else cur_cert_frac.item(),
      "cert_eps": cur_cert_eps if isinstance(cur_cert_eps, float) else cur_cert_eps.item(),
      "epoch": t,
    }
    with tqdm(data) as batch_loop:
      for batch_idx, batch in enumerate(batch_loop):
        batch = data_util.dict_batch_to_device(batch, device)
        optimizer.zero_grad()
        if cur_cert_frac > 0.0:
          out = model.forward(batch, cert_eps=cur_cert_eps)
          logits = out.val
          loss = loss_func(logits, batch['y'])
          epoch["clean_loss"] += loss.item()
          cert_loss = torch.max(loss_func(out.lb, batch['y']),
                               loss_func(out.ub, batch['y']))
          loss = cur_cert_frac * cert_loss + (1.0 - cur_cert_frac) * loss
          epoch["cert_loss"] += cert_loss.item()
        else:
          # Bypass computing bounds during training
          logits = out = model.forward(batch, compute_bounds=False)
          loss = loss_func(logits, batch['y'])
        epoch["total_loss"] += loss.item()
        epoch["num"] += len(batch['y'])
        num_correct, num_cert_correct = task_class.num_correct(out, batch['y'])
        epoch["num_correct"] += num_correct
        epoch["num_cert_correct"] += num_cert_correct
        loss.backward()
        if any(p.grad is not None and torch.isnan(p.grad).any() for p in model.parameters()):
          nan_params = [p.name for p in model.parameters() if p.grad is not None and torch.isnan(p.grad).any()]
          print('NaN found in gradients: %s' % nan_params, file=sys.stderr)
        else:
          if clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
          optimizer.step()
      if cert_frac > 0.0:
        print("Epoch {epoch:>3}: train loss: {total_loss:.6f}, clean_loss: {clean_loss:.6f}, cert_loss: {cert_loss:.6f}".format(**epoch))
      else:
        print("Epoch {epoch:>3}: train loss: {total_loss:.6f}".format(**epoch))
      sys.stdout.flush()

    epoch["clean_acc"] = 100.0 * epoch["num_correct"] / epoch["num"]
    acc_str = "  Train accuracy: {num_correct}/{num} = {clean_acc:.2f}".format(**epoch)
    if cert_frac > 0.0:
      epoch["cert_acc"] = 100.0 * epoch["num_cert_correct"] / epoch["num"]
      acc_str += ", certified {num_cert_correct}/{num} = {cert_acc:.2f}".format(**epoch)
    print(acc_str)
    is_best = False
    if dev_data:
      dev_results = test(task_class, model, "Dev", dev_data, device, batch_size=batch_size,
                         aug_dataset=aug_dev_data)
      epoch['dev'] = dev_results
      all_epoch_stats['acc']['dev']['clean'].append(dev_results['clean_acc'])
      all_epoch_stats['acc']['dev']['cert'].append(dev_results['cert_acc'])
      if augmenter:
        all_epoch_stats['acc']['dev']['aug'].append(dev_results['aug_acc'])
      dev_stats = {
          'epoch': t, 
          'loss': dev_results['loss'], 
          'clean_acc': dev_results['clean_acc'], 
          'cert_acc': dev_results['cert_acc']
      }
      if augmenter:
        dev_stats['aug_acc'] = dev_results['aug_acc']
      if dev_results['clean_acc'] > all_epoch_stats['acc']['best_dev']['clean'][-1]['clean_acc']:
        all_epoch_stats['acc']['best_dev']['clean'].append(dev_stats)
        if cert_frac == 0.0 and not augmenter:
          is_best = True
      if dev_results['cert_acc'] > all_epoch_stats['acc']['best_dev']['cert'][-1]['cert_acc']:
        all_epoch_stats['acc']['best_dev']['cert'].append(dev_stats)
        if cert_frac > 0.0:
          is_best = True
      if augmenter and dev_results['aug_acc'] > all_epoch_stats['acc']['best_dev']['aug'][-1]['aug_acc']:
        all_epoch_stats['acc']['best_dev']['aug'].append(dev_stats)
        if cert_frac == 0.0 and augmenter:
          is_best = True
      epoch['best_dev'] = {
              'clean': all_epoch_stats['acc']['best_dev']['clean'][-1],
              'cert': all_epoch_stats['acc']['best_dev']['cert'][-1]}
      if augmenter:
        epoch['best_dev']['aug'] = all_epoch_stats['acc']['best_dev']['aug'][-1]
    all_epoch_stats["loss"]['total'].append(epoch["total_loss"])
    all_epoch_stats["loss"]['clean'].append(epoch["clean_loss"])
    all_epoch_stats["loss"]['cert'].append(epoch["cert_loss"])
    all_epoch_stats['cert']['frac'].append(epoch["cert_frac"])
    all_epoch_stats['cert']['eps'].append(epoch["cert_eps"])
    all_epoch_stats["acc"]['train']['clean'].append(epoch["clean_acc"])
    all_epoch_stats["acc"]['train']['cert'].append(epoch["cert_acc"])
    with open(os.path.join(OPTS.out_dir, "run_stats.json"), "w") as outfile:
      json.dump(epoch, outfile)
    with open(os.path.join(OPTS.out_dir, "all_epoch_stats.json"), "w") as outfile:
      json.dump(all_epoch_stats, outfile)
    if ((save_best_only and is_best) 
        or (not save_best_only and epochs_per_save and (t+1) % epochs_per_save == 0)
        or t == num_epochs - 1):
      if save_best_only and is_best:
        for fn in glob.glob(os.path.join(OPTS.out_dir, 'model-checkpoint*.pth')):
          os.remove(fn)
      model_save_path = os.path.join(OPTS.out_dir, "model-checkpoint-{}.pth".format(t))
      print('Saving model to %s' % model_save_path)
      torch.save(model.state_dict(), model_save_path)

  return model


def test(task_class, model, name, dataset, device, show_certified=False, batch_size=1,
         adversary=None, aug_dataset=None):
  model.eval()
  loss_func = task_class.LOSS_FUNC
  results = {
      'name': name,
      'num_total': 0,
      'num_correct': 0,
      'num_cert_correct': 0,
      'clean_acc': 0.0,
      'cert_acc': 0.0,
      'loss': 0.0
  }
  data = dataset.get_loader(batch_size)
  with torch.no_grad():
    for batch in tqdm(data):
      batch = data_util.dict_batch_to_device(batch, device)
      out = model.forward(batch, cert_eps=1.0)
      results['loss'] += loss_func(out.val, batch['y']).item()
      num_correct, num_cert_correct = task_class.num_correct(out, batch['y'])
      results["num_correct"] += num_correct
      results["num_cert_correct"] += num_cert_correct
      results['num_total'] += len(batch['y'])
    if aug_dataset:
      results['aug_loss'] = results['loss']
      results['aug_total'] = results['num_total']
      results['aug_correct'] = results['num_correct']
      aug_data = aug_dataset.get_loader(batch_size)
      for batch in tqdm(aug_data):
        batch = data_util.dict_batch_to_device(batch, device)
        out = model.forward(batch, cert_eps=1.0)
        results['aug_loss'] += loss_func(out.val, batch['y']).item()
        num_correct, num_cert_correct = task_class.num_correct(out, batch['y'])
        results["aug_correct"] += num_correct
        results['aug_total'] += len(batch['y'])
  results['clean_acc'] = 100 * results['num_correct'] / results['num_total']
  results['cert_acc'] = 100 * results['num_cert_correct'] / results['num_total']
  out_str = "  {name} loss = {loss:.2f}; accuracy: {num_correct}/{num_total} = {clean_acc:.2f}, certified {num_cert_correct}/{num_total} = {cert_acc:.2f}".format(**results)
  if aug_dataset:
    results['aug_acc'] = 100 * results['aug_correct'] / results['aug_total']
    out_str += ', augmented %d/%d = %.2f' % (
        results['aug_correct'], results['aug_total'], results['aug_acc'])
  if adversary:
    adv_correct, adv_exs = adversary.run(model, dataset, device, opts=OPTS)
    results['num_adv_correct'] = sum(adv_correct)
    results['adv_acc'] = 100 * results['num_adv_correct'] / len(dataset)
    out_str += ', adversarial %d/%d = %.2f' % (
        results['num_adv_correct'], len(dataset), results['adv_acc'])
  print(out_str)
  return results


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('task', choices=TASK_CLASSES.keys())
  parser.add_argument('model', choices=['bow', 'cnn', 'lstm', 'decomp-attn', 'lstm-final-state'])
  parser.add_argument('out_dir', help='Directory to store and load output')
  # Model
  parser.add_argument('--hidden-size', '-d', type=int, default=100)
  parser.add_argument('--kernel-size', '-k', type=int, default=3,
                      help='Kernel size, for CNN convolutions and pooling')
  parser.add_argument('--pool', choices=['max', 'mean', 'attn'], default='max')
  parser.add_argument('--num-layers', type=int, default=3, help='Num layers for SNLI baseline BOW model')
  parser.add_argument('--no-wordvec-layer', action='store_true', help="Don't apply linear transform to word vectors")
  parser.add_argument('--early-ibp', action='store_true', help="Do to_interval_bounded directly on base word vectors")
  parser.add_argument('--no-relu-wordvec', action='store_true', help="Don't do ReLU after word vector transform")
  parser.add_argument('--unfreeze-wordvec', action='store_true', help="Don't freeze word vectors")
  parser.add_argument('--glove', '-g', choices=vocabulary.GLOVE_CONFIGS, default='840B.300d')
  # Adversary
  parser.add_argument('--adversary', '-a', choices=['exhaustive', 'greedy', 'genetic'],
                      default=None, help='Which adversary to test on')
  parser.add_argument('--adv-num-epochs', type=int, default=10)
  parser.add_argument('--adv-num-tries', type=int, default=2)
  parser.add_argument('--adv-pop-size', type=int, default=60)
  parser.add_argument('--use-lm', action='store_true', help='Use LM scores to define attack surface')
  # Training
  parser.add_argument('--num-epochs', '-T', type=int, default=1)
  parser.add_argument('--learning-rate', '-r', type=float, default=1e-3)
  parser.add_argument('--dropout-prob', type=float, default=0.1)
  parser.add_argument('--batch-size', '-b', type=int, default=1)
  parser.add_argument('--clip-grad-norm', type=float, default=0.25)
  parser.add_argument('--weight-decay', type=float, default=1e-4)
  parser.add_argument('--cert-frac', '-c', type=float, default=0.0,
                      help='Fraction of loss devoted to certified loss term.')
  parser.add_argument('--initial-cert-frac', type=float, default=0.0,
                      help='If certified loss is being used, where the linear scale for it begins')
  parser.add_argument('--cert-eps', type=float, default=1.0,
                      help='Max scaling factor for the interval bounds of the attack words to be used')
  parser.add_argument('--initial-cert-eps', type=float, default=0.0,
                      help='If certified loss is being used, where the linear scale for its epsilon begins')
  parser.add_argument('--full-train-epochs', type=int, default=0,
                      help='If specified use full cert_frac and cert_eps for this many epochs at the end')
  parser.add_argument('--non-cert-train-epochs', type=int, default=0,
                      help='If specified train this many epochs regularly in beginning')
  parser.add_argument('--epochs-per-save', type=int, default=1,
                      help='How often to save model; 0 to only save final model')
  parser.add_argument('--save-best-only', action='store_true',
                      help='Only save best dev epochs (based on cert acc if cert_frac > 0, clean acc else)')
  parser.add_argument('--augment-by', type=int, default=0,
                      help='How many augmented examples per real example')
  # Data and files
  parser.add_argument('--adv-only', action='store_true', help='Only run the adversary against the model on the given evaluation set')
  parser.add_argument('--test', action='store_true', help='Evaluate on test set')
  parser.add_argument('--data-cache-dir', '-D', help='Where to load cached dataset and glove')
  parser.add_argument('--neighbor-file', type=str, default=data_util.NEIGHBOR_FILE)
  parser.add_argument('--glove-dir', type=str, default=vocabulary.GLOVE_DIR)
  parser.add_argument('--imdb-dir', type=str, default=text_classification.IMDB_DIR)
  parser.add_argument('--snli-dir', type=str, default=entailment.SNLI_DIR)
  parser.add_argument('--imdb-lm-file', type=str, default=text_classification.LM_FILE)
  parser.add_argument('--snli-lm-file', type=str, default=entailment.LM_FILE)
  parser.add_argument('--prepend-null', action='store_true', help='If true add UNK token to sequences')
  parser.add_argument('--normalize-word-vecs', action='store_true', help='If true normalize word vectors')
  parser.add_argument('--downsample-to', type=int, default=None,
                      help='Downsample train and dev data to this many examples')
  parser.add_argument('--downsample-shard', type=int, default=0,
                      help='Downsample starting at this multiple of downsample_to')
  parser.add_argument('--use-toy-data', action='store_true')
  parser.add_argument('--truncate-to', type=int, default=None,
                      help='Truncate examples to this max length')
  # Loading
  parser.add_argument('--load-dir', '-L', help='Where to load checkpoint')
  parser.add_argument('--load-ckpt', type=int, default=None,
                      help='Which checkpoint to load')
  # Other
  parser.add_argument('--rng-seed', type=int, default=123456)
  parser.add_argument('--torch-seed', type=int, default=1234567)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()


def main():
  random.seed(OPTS.rng_seed)
  np.random.seed(OPTS.rng_seed)
  torch.manual_seed(OPTS.torch_seed)
  torch.backends.cudnn.deterministic = True
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  task_class = TASK_CLASSES[OPTS.task]
  print('Loading dataset.')
  if not os.path.exists(OPTS.out_dir):
    os.makedirs(OPTS.out_dir)
  with open(os.path.join(OPTS.out_dir, 'log.txt'), 'w') as f:
    print(sys.argv, file=f)
    print(OPTS, file=f)
  if OPTS.data_cache_dir:
    if not os.path.exists(OPTS.data_cache_dir):
        os.makedirs(OPTS.data_cache_dir)
  train_data, dev_data, word_mat, attack_surface = task_class.load_datasets(device, OPTS)
  print('Initializing model.')
  model = task_class.load_model(word_mat, device, OPTS)
  if OPTS.num_epochs > 0:
    augmenter = None
    if OPTS.augment_by:
      augmenter = task_class.DataAugmenter(OPTS.augment_by)
    train(task_class, model, train_data, OPTS.num_epochs, OPTS.learning_rate, device,
          dev_data=dev_data, cert_frac=OPTS.cert_frac, initial_cert_frac=OPTS.initial_cert_frac,
          cert_eps=OPTS.cert_eps, initial_cert_eps=OPTS.initial_cert_eps, batch_size=OPTS.batch_size,
          epochs_per_save=OPTS.epochs_per_save, augmenter=augmenter, clip_grad_norm=OPTS.clip_grad_norm,
          weight_decay=OPTS.weight_decay, full_train_epochs=OPTS.full_train_epochs, non_cert_train_epochs=OPTS.non_cert_train_epochs, save_best_only=OPTS.save_best_only)
    print('Training finished.')
  print('Testing model.')
  if not OPTS.adv_only:
    train_results = test(task_class, model, 'Train', train_data, device, 
                         batch_size=OPTS.batch_size)
    adversary = None
    if OPTS.adversary == 'exhaustive':
      adversary = task_class.ExhaustiveAdversary(attack_surface)
    elif OPTS.adversary == 'greedy':
      adversary = task_class.GreedyAdversary(attack_surface, num_epochs=OPTS.adv_num_epochs,
                                             num_tries=OPTS.adv_num_tries)
    elif OPTS.adversary == 'genetic':
      adversary = task_class.GeneticAdversary(attack_surface, num_iters=OPTS.adv_num_epochs,
                                              pop_size=OPTS.adv_pop_size)
    dev_results = test(task_class, model, 'Dev', dev_data, device, 
                       adversary=adversary, batch_size=OPTS.batch_size)
    results = {
        'train': train_results,
        'dev': dev_results
    }
    with open(os.path.join(OPTS.out_dir, 'test_results.json'), 'w') as f:
      json.dump(results, f)
  else:
    adversary = None
    if OPTS.adversary == 'exhaustive':
      adversary = task_class.ExhaustiveAdversary(attack_surface)
    elif OPTS.adversary == 'greedy':
      adversary = task_class.GreedyAdversary(attack_surface, num_epochs=OPTS.adv_num_epochs,
                                             num_tries=OPTS.adv_num_tries)
    elif OPTS.adversary == 'genetic':
      adversary = task_class.GeneticAdversary(attack_surface, num_iters=OPTS.adv_num_epochs,
                                              pop_size=OPTS.adv_pop_size)
    test(task_class, model, 'Dev', dev_data, device, adversary=adversary, batch_size=OPTS.batch_size)

if __name__ == '__main__':
  OPTS = parse_args()
  main()
