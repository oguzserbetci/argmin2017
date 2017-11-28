# argmin2017
Argumentation Mining Project, Uni Potsdam, WS17/18

## Data
We use microtext corpus [1] from [`peldszus/arg-microtexts`](https://github.com/peldszus/arg-microtexts).

## Architecture
We use an architecture as described in [2].

## Setup
Pipenv is used for dependencies and for virtual environment. Setting up a new python 3 virtual environment with `pipenv` is easy:

```bash
brew install pipenv
pipenv --three
pipenv install
```

Enter the virtual environment with `pipenv source` command.

# References
[1] Andreas Peldszus, Manfred Stede. [An annotated corpus of argumentative microtexts. First European Conference on Argumentation: Argumentation and Reasoned Action](http://www.ling.uni-potsdam.de/%7Epeldszus/eca2015-preprint.pdf), Portugal, Lisbon, June 2015.

[2] Peter Potash, Alexey Romanov, Anna Rumshisky 2017, [arXiv:1612.08994](https://arxiv.org/abs/1612.08994)
