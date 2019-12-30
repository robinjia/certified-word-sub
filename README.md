# Certified Robustness to Adversarial Word Substitutions

This is the official GitHub repository for the following paper:

> [**Certified Robustness to Adversarial Word Substitutions.**](https://arxiv.org/abs/1909.00986)  
> Robin Jia, Aditi Raghunathan, Kerem Göksel, and Percy Liang.  
> _Empirical Methods in Natural Language Processing (EMNLP)_, 2019.  

For full details on reproducing the results, see this [Codalab worksheet](https://worksheets.codalab.org/worksheets/0x79feda5f1998497db75422eca8fcd689),
which contains all code, data, and experiments from the paper.
This GitHub repository serves as an easy way to get started with the code, and has some additional instructions and documentation.

# Setup
This code has been tested with python3.6, pytorch 1.3.1, numpy 1.15.4, and NLTK 3.4.

Download data dependencies by running the provided script:
```
./download_deps.sh
```

If you already have GloVe vectors on your system, 
it may be more convenient to comment out the part of `download_deps.sh` that downloads GloVe,
and instead add a symlink to the directory containing the GloVe vectors at `data/glove`.

# Interval Bound Propagation library

We have implemented many primitives for Interval Bound Propagation (IBP),
which can be found in `src/ibp.py`. This code should be reusable and intuitive for anyone familiar with pytorch.
When designing this library, our goal was to make it possible to write code that looks like standard pytorch code, but can be trained with IBP.
Below, we give an overview of the code.

## BoundedTensor
`BoundedTensor` is our version of `torch.Tensor`. It represents a tensor that additionally has some bounded set of possible values. The two most important subclasses of `BoundedTensor` are `IntervalBoundedTensor` and `DiscreteChoiceTensor`.

### IntervalBoundedTensor
An `IntervalBoundedTensor` keeps track of three instance variables: an actual value, a coordinate-wise upper bound on the value, and a coordinate-wise lower bound on the value. All three of these are `torch.Tensor` objects.
It also implements many standard methods of `torch.Tensor`.

### DiscreteChoiceTensor
A `DiscreteChoiceTensor` represents a tensor that can take a discrete set of values.
We use `DiscreteChoiceTensor` to represent the set of possible word vectors
that can appear at each slice of the input.
Importantly, `DiscreteChoiceTensor.to_interval_bounded()` converts a `DiscreteChoiceTensor` to an `IntervalBoundedTensor` by taking a coordinate-wise min/max.

### NormBallTensor
We also provide `NormBallTensor`, which represents a p-norm ball of a given radius around a value.

## Functions and layers
To go with `BoundedTensor`, we include functions and layers that know how to take `BoundedTensor` objects as inputs
and return `BoundedTensor` objects as outputs.
Most of these should be straightforward to use for folks familiar with their standard `torch`, `torch.nn`, and `torch.nn.functional` equivalents (with a caveat that not all flags in the standard library are necessarily supported).

### Functions
Available implementations of basic `torch` functions include:

* `add`
* `mul`
* `div`
* `bmm`
* `cat`
* `stack`
* `sum`

In many cases, we directly call the `torch` counterpart if the inputs are `torch.Tensor` objects.
A few additional cases are described below.

#### Activation functions
Since monotonic functions all use the same IBP formula, 
we export a single function `ibp.activation`
which can apply elementwise ReLU, sigmoid, tanh, or exp to an `IntervalBoundedTensor`.

#### Logsoftmax
We include a `log_softmax()` function that is equivalent to `torch.nn.functional.log_softmax()`.
We strongly advise users to use this implementation rather than implementing their own softmax operation, as numerical instability can easily arise with a naive implementation.

#### Nonnegative matrix multiplication
We include `matmul_nneg()` function that handles matrix multiplication between two non-negative matrices, as this is simpler than the general case.

### Layers (`nn.Module` objects)
Many basic layers are implemented by extending their `torch.nn` counterparts, including
* `Linear` 
* `Embedding`
* `Conv1d`
* `MaxPool1d`
* `LSTM`
* `Dropout`

#### RNNs 
Our library also includes `LSTM` and `GRU` classes, which extend `nn.Module` directly.
These are unfortunately slower than their `torch.nn` counterparts,
because the `torch.nn` RNN's use cuDNN.

## Examples
If you want to see this library in action, a good place to start is `BOWModel` in `src/text_classification.py`. This implements a simple bag-of-words model for text classification.
Note that in `forward()`, we accept a flag called `compute_bounds` which lets the user decide whether to run IBP or not.

# Paper experiments
In this repository, we include a minimal set of commands and instructions to reproduce a few key results from our EMNLP 2019 paper.
We will focus on the CNN model results on the IMDB dataset.
To see other available command line flags, you can run `python src/train.py -h`.

If you are interested in reproducing our experiments, we recommend looking at the
aforementioned [Codalab worksheet](https://worksheets.codalab.org/worksheets/0x79feda5f1998497db75422eca8fcd689), which shows how to reproduce all results in our paper.
Note that the commands on Codalab include some extra flags (`--neighbor-file`, `--glove-dir`, `--imdb-dir`, and `--snli-dir`) that are used to specify non-default paths to files.
These flags are unnecessary when following the instructions in this repository.

## Training
Here are commands to train the CNN model on IMDB with standard training, certifiably robust training, and data augmentation.

**Standard training**

To train the baseline model without IBP, run the following:
```
python src/train.py classification cnn outdir_cnn_normal -d 100 --pool mean -T 10 --dropout-prob 0.2 -b 32 --save-best-only
```

This should get about 88% accuracy on dev (but 0% certified accuracy).
`outdir_cnn_normal` is an output directory where model parameters and stats will be saved.

**Certifiably robust training**

To use certifiably robust training with IBP, run the following:
```
python src/train.py classification cnn outdir_cnn_cert -d 100 --pool mean -T 60 --full-train-epochs 20 -c 0.8 --dropout-prob 0.2 -b 32 --save-best-only
```

This should get about 81% accuracy and 66% certified accuracy on dev.
Note that these results do not include language model constraints on the attack surface,
and therefore the certified accuracy is a bit too low.
These constraints will be enforced in the testing commands below.

**Training with data augmentation**

To train with data augmentation, run the following:
```
python src/train.py classification cnn outdir_cnn_aug -d 100 --pool mean -T 60 --augment-by 4 --dropout-prob 0.2 -b 32 --save-best-only
```

This should get about 85% accuracy and 84% augmented accuracy on dev (but 0% certified accuracy).

## Testing
Next, we will show how to test the trained models using the genetic attack.
The genetic attack heuristically searches for a perturbation that causes an error.
In this phase, we also incorporate pre-computed language model scores that determine which perturbations are valid.

For example, let's say we want to use the trained model inside the `outdir_cnn_cert` directory.
First, we choose a checkpoint based on the best certified accuracy on the dev set, say checkpoint 57.
(Note: the training code with `--save-best-only` will save only the best model and the final model;
stats on all checkpoints are logged in `<outdir>/all_epoch_stats.json`.)

This command will run the genetic attack:
```
python src/train.py classification cnn eval_cnn_cert -L outdir_cnn_cert --load-ckpt 57 -d 100 --pool mean -T 0 -b 1 -a genetic --adv-num-epochs 40 --adv-pop-size 60 --use-lm --downsample-to 1000
```
It should get about 80% standard accuracy, 72.5% certified accuracy, 
and 73% adversarial accuracy (i.e., accuracy against the genetic attack).
For all models, you should find that adversarial accuracy is between standard accuracy and certified accuracy.
For IMDB, we downsample to 1000 examples, as the genetic attack is pretty slow;
the provided precomputed LM scores (in `lm_scores`) are only for the first 1000 examples in the train, development, and test sets.
For SNLI, we use the entire development and test sets for evaluation.

**Note:** This code is sensitive to the version of NLTK you use.
The LM prediction files provided here should work if you are using the current version of NLTK and have updated your `nltk_data` directory recently. 
The experiments on Codalab use an older NLTK version;
you can download the LM files from Codalab if you need compatibility with older NLTK versions.
NLTK version issues will result in a `KeyError` with an `Unrecognized sentence` message.

## Running the language model yourself
If you want to precompute language model scores on other data, use the following instructions.

1. Clone the following git repository:

```
git clone https://github.com/robinjia/l2w windweller-l2w
```

2. Obtain pre-trained parameters and put them in a 
directory named `l2w-params` within that repository.
Please contact us if you need a copy of the parameters. 

3. Adapt `src/precompute_lm_scores.py` for your dataset.
