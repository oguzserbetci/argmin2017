# argmin2017
Argumentation Mining Project, Uni Potsdam, WS17/18

This repository includes code that is implementing and reproducing results from Potash et al. [3].

## Data
We use microtext corpus [1] from [`peldszus/arg-microtexts`](https://github.com/peldszus/arg-microtexts).

## Architecture
We use a seq-to-seq pointer network [2] architecture for argument mining as described in [3].

## Setup
Pipenv is used for dependencies and for virtual environment. Setting up a new python 3 virtual environment with `pipenv` is easy:

```bash
brew install pipenv
pipenv install --three [--dev] # use --dev for embedding the corpus later
```

Enter the virtual environment with `pipenv source`.

## Training
First, you need to create the embeddings for the microtext corpus [1]:

```
pipenv shell
cd corpus
python corpus.py -ei encoder_input -di decoder_input -l links -t types -p 'arg-microtexts/corpus/en/*.xml'
```

Now you can train the network with keras as follows:

```
pipenv shell
cd keras
python train.py -ei ../corpus/encoder_input.npy -di ../corpus/decoder_input.npy -l ../corpus/links.npy -t ../corpus/types.npy -e 4000

# If your machine has GPUS and you do not want to use:
CUDA_VISIBLE_DEVICES=-1 python train.py -ei ../corpus/encoder_input.npy -di ../corpus/decoder_input.npy -l ../corpus/links.npy -t ../corpus/types.npy -e 4000

# If your machine has GPUS and you want to use all:
python train.py -ei ../corpus/encoder_input.npy -di ../corpus/decoder_input.npy -l ../corpus/links.npy -t ../corpus/types.npy -e 4000 -g 4

# If your machine has GPUS but you only want to use a specific one, for example the first two:
CUDA_VISIBLE_DEVICES=0,1 python train.py -ei ../corpus/encoder_input.npy -di ../corpus/decoder_input.npy -l ../corpus/links.npy -t ../corpus/types.npy -e 4000 -g 2
```

## Results
Results are written in `cross_validation/` you can inspect it with the jupyter notebooks in `notebooks/`

## NOTE
pytorch implementation is **not complete** and not tested.

# References
[1] Peldszus, Stede 2016 [An annotated corpus of argumentative microtexts](http://www.ling.uni-potsdam.de/%7Epeldszus/eca2015-preprint.pdf)

[2] Vinyals, Fortunato, Jaitly 2015 [Pointer Networks](https://arxiv.org/abs/1612.08994)

[3] Potash, Romanov, Rumshisky 2017 [Here's my Point: Joint Pointer Architecture for Argument Mining](https://arxiv.org/abs/1612.08994)
