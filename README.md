[![github](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/jwieckowski/pysensmcda)
[![DOI:10.1016/j.softx.2022.101271](http://img.shields.io/badge/DOI-10.1016/j.softx.2022.101271-0f81c2.svg)](https://doi.org/10.1016/j.softx.2022.101271)
[![Version](https://img.shields.io/pypi/v/pyfdm)](https://pypi.org/project/)

<!-- [![License](https://img.shields.io/github/license/nlesc/pyfdm)]() -->

## PySensMCDA

`PySensMCDA` is a comprehensive Python package tailored specifically for Multi-Criteria Decision Analysis (MCDA) sensitivity analysis. MCDA is a powerful tool used in decision-making processes to evaluate alternatives based on multiple conflicting criteria. `PySensMCDA` empowers users to delve deeper into the robustness and reliability of their decision models by exploring the sensitivity of results to variations in input parameters.

In essence, this package offers tools for:

- Decision matrix sensitivity analysis
- Weights sensitivity analysis
- Ranking sensitivity analysis
- Perturbation generation
- Weights generation
- Visualizations of sensitivity analysis

## Installation

The package can be download using pip:

```Bash
pip install pysensmcda
```

## Testing

The modules performance can be verified with `pytest` library

```Bash
pip install pytest
pytest tests
```

<!--
## Citation

If you use `PySensMCDA` in you work, please cite the following [publication](https://scholar.google.pl/):

> Citation

As BibTeX:

```
citation
```
-->

## Modules and functionalities

<br/>

- ### Alternative:

| Name                    |  Reference   |
| ----------------------- | :----------: |
| Discrete modification   | [[0]](#ref0) |
| Percentage modification | [[0]](#ref0) |
| Range modification      | [[0]](#ref0) |
| Alternative removal     | [[0]](#ref0) |

<br/>

- ### Criteria:

| Name                                             |  Reference   |
| ------------------------------------------------ | :----------: |
| Random distribution - weights generation         |      -       |
| &nbsp;&nbsp;&nbsp;&nbsp; Chisquare distribution  |      -       |
| &nbsp;&nbsp;&nbsp;&nbsp; Laplace distribution    |      -       |
| &nbsp;&nbsp;&nbsp;&nbsp; Normal distribution     |      -       |
| &nbsp;&nbsp;&nbsp;&nbsp; Random distribution     |      -       |
| &nbsp;&nbsp;&nbsp;&nbsp; Triangular distribution |      -       |
| &nbsp;&nbsp;&nbsp;&nbsp; Uniform distribution    |      -       |
| Percentage modification                          | [[0]](#ref0) |
| Range modification                               | [[0]](#ref0) |
| Weights scenarios                                | [[0]](#ref0) |
| Cirteria identification                          | [[0]](#ref0) |
| Criteria removal                                 | [[0]](#ref0) |

<br/>

- ### Probabilistic:

| Name                           |  Reference   |
| ------------------------------ | :----------: |
| Monte carlo weights generation | [[0]](#ref0) |
| Perturbated matrix             | [[0]](#ref0) |
| Perturbated weights            | [[0]](#ref0) |

<br/>

- ### Ranking:

| Name               |  Reference   |
| ------------------ | :----------: |
| Ranking alteration | [[0]](#ref0) |
| Demotion           | [[0]](#ref0) |
| Promotion          | [[0]](#ref0) |
| Fuzzy ranking      | [[0]](#ref0) |

<br/>

- ### Compromise:

| Name                                         |  Reference   |
| -------------------------------------------- | :----------: |
| Borda                                        | [[0]](#ref0) |
| Improved Borda                               | [[0]](#ref0) |
| Dominance directed graph                     | [[0]](#ref0) |
| Half-quadratic compromise                    | [[0]](#ref0) |
| ICRA - Iterative Compromise Ranking Analysis | [[0]](#ref0) |
| Rank position method                         | [[0]](#ref0) |

<br/>

- ### Graphs:

| Name                             |
| -------------------------------- |
| Heatmap                          |
| Promotion-demotion ranking graph |
| Preference distribution          |
| Rankings distribution            |
| Values distribution              |
| Weights barplot                  |

## Usage example

- General usage examples see [`examples.ipynb`](./examples/examples.ipynb)
- Graphs submodule usage examples see [`graphs_examples.ipynb`](./examples/graphs_examples.ipynb)
- Literature example analysis see [`literature_example.ipynb`](./examples/literature_example.ipynb)

## Related work

Don't forget to check out these other amazing software packages!

- [Make-Decision.it](http://make-decision.it/): Web application offering users a graphical interface for prototyping structural decision models
- [PyFDM](https://pypi.org/project/pyfdm/): package with Fuzzy Decision Making (PyFDM) methods based on Triangular Fuzzy Numbers (TFN).
- [PyIFDM](https://pypi.org/project/pyifdm/): package to perform Multi-Criteria Decision Analysis in the Intuitionistic Fuzzy environment.
- [PyMCDM](https://pypi.org/project/pymcdm/): Python 3 library for solving multi-criteria decision-making (MCDM) problems.

### References
