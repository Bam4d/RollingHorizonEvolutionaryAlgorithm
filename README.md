# Rolling Horizon Evolutionary Algorithm

[![PyPI version](https://badge.fury.io/py/RollingHorizonEA.svg)](https://badge.fury.io/py/RollingHorizonEA)

An implementation of the [Rolling Horizon Evolutionary Algorithm](https://www.semanticscholar.org/paper/Rolling-horizon-evolution-versus-tree-search-for-in-Liebana-Samothrakis/0cff838805be4b6366756a553daca0036778c1e0)

## Installation

### using pip

```
pip install RollingHorizonEA
```

## Usage

To use the rolling horizon evolutionary algorithm, you will need your game class to implement the `Environment` interface.

### Examples

Examples of setting up any game environment can be found in the `examples` directory and run with:
```
python run.py
```

#### m_max example

```
num_dims = 600
m = 50
num_evals = 50
rollout_length = 10
mutation_probability = 0.1

# Set up the problem domain as m-max game
environment = MMaxGame(num_dims, m)

rhea = RollingHorizonEvolutionaryAlgorithm(rollout_length, environment, mutation_probability, num_evals)

rhea.run()
```


## Cite

If you want to cite this library, please use the following DOI

[![DOI](https://zenodo.org/badge/172040305.svg)](https://zenodo.org/badge/latestdoi/172040305)
