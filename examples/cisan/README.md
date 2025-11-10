<p align="center">
    <h1 align="center">CISAN</h1>
</p>

<p align="center">
	<img src="https://img.shields.io/github/license/pckennethma/CISan?style=flat&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/pckennethma/CISan?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/pckennethma/CISan?style=flat&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/pckennethma/CISan?style=flat&color=0080ff" alt="repo-language-count">
<p>
<p align="center">
		<em>This is a research artifact for paper "Enabling Runtime Verification of Causal Discovery Algorithms with Automated Conditional Independence Reasoning" (ICSE 2024).</em>
</p>
<p align="center">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
</p>
<hr>

## 📍 Overview

Welcome to the repository for CISan, a suite of tools and utilities designed for runtime verification of causal discovery algorithms through automated conditional independence reasoning. This repository contains the Python implementation and dataset resources necessary for conducting experiments and validating the performance of various causal discovery algorithms.


## 📂 Repository Structure

```sh
└── CISan/
    ├── Chisq.py
    ├── DataUtils.py
    ├── EDSanPC.py
    ├── GraphUtils.py
    ├── IndependenceSolver.py
    ├── Kendall.py
    ├── LICENSE
    ├── OpAblation.py
    ├── PC.py
    ├── ParallelSolver.py
    ├── README.md
    ├── Rules.py
    ├── Utility.py
    └── data
        ├── alarm-10k.csv
        ├── cancer-10k.csv
        ├── earthquake-10k.csv
        ├── insurance-10k.csv
        ├── sachs-10k.csv
        └── survey-10k.csv
```


## 🧩 Modules


| File                                                                                            | Summary                                                                                   |
| ---                                                                                             | ---                                                                                       |
| [OpAblation.py](https://github.com/pckennethma/CISan/blob/master/OpAblation.py)                 | Contains ablation studies for evaluating the impact of various components in the algorithms. |
| [Rules.py](https://github.com/pckennethma/CISan/blob/master/Rules.py)                           | Encodes Pearl's axioms and other rules used in conditional independence reasoning.         |
| [Kendall.py](https://github.com/pckennethma/CISan/blob/master/Kendall.py)                       | Implements Kendall’s tau test for assessing conditional independence.                      |
| [EDSanPC.py](https://github.com/pckennethma/CISan/blob/master/EDSanPC.py)                       | Executes causal discovery algorithms with runtime verification for error detection.        |
| [ParallelSolver.py](https://github.com/pckennethma/CISan/blob/master/ParallelSolver.py)         | Provides parallelized solvers to enhance performance of conditional independence testing.  |
| [Utility.py](https://github.com/pckennethma/CISan/blob/master/Utility.py)                       | General utility functions used across different modules.                                   |
| [GraphUtils.py](https://github.com/pckennethma/CISan/blob/master/GraphUtils.py)                 | Functions for graph manipulation and visualization.                                        |
| [Chisq.py](https://github.com/pckennethma/CISan/blob/master/Chisq.py)                           | Implements chi-squared tests for evaluating conditional independence.                      |
| [DataUtils.py](https://github.com/pckennethma/CISan/blob/master/DataUtils.py)                   | Utility functions for data preprocessing and manipulation.                                  |
| [IndependenceSolver.py](https://github.com/pckennethma/CISan/blob/master/IndependenceSolver.py) | Core solver for conditional independence reasoning using SMT.                              |
| [PC.py](https://github.com/pckennethma/CISan/blob/master/PC.py)                                 | Implementation of the PC algorithm for causal discovery.                                   |



## 🚀 Getting Started

***Requirements***

Ensure you have the following dependencies installed on your system:

* **Python**: `version 3.9.7`

### ⚙️ Installation

1. Clone the CISan repository:

```sh
git clone https://github.com/pckennethma/CISan
```

2. Change to the project directory:

```sh
cd CISan
```

3. Install the dependencies:

Currently, `requirements.txt` is not available. We will prepare it shortly. Apologize for any inconvenience!

```sh
pip install -r requirements.txt
```

### 🤖 Running CISan

Use the following command to run CISan:

```
python PC.py # run P-CISan
python EDSanPC.py # run ED-CISan
python OpAblation.py # run ablation experiments
```

## 🤝 Contributing

We welcome contributions from the community. If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request. For major changes, please discuss them via an issue first to ensure a smooth collaboration.

## 📚 Citation

If you find this repository useful in your research, please cite our paper:

```bibtex
@inproceedings{ma2024enabling,
  title={Enabling Runtime Verification of Causal Discovery Algorithms with Automated Conditional Independence Reasoning},
  author={Ma, Pingchuan and Ji, Zhenlan and Yao, Peisen and Wang, Shuai and Ren, Kui},
  booktitle={Proceedings of the 46th IEEE/ACM International Conference on Software Engineering},
  pages={1--13},
  year={2024}
}
```

For more details, you can access the full paper [here](https://doi.org/10.1145/3597503.3623348).

Feel free to reach out if you have any questions or need further assistance regarding the citation or usage of this repository.
