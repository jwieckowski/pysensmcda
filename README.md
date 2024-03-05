[![github](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/jwieckowski/pysensmcda)
[![DOI:10.1016/j.softx.2022.101271](http://img.shields.io/badge/DOI-10.1016/j.softx.2022.101271-0f81c2.svg)](https://doi.org/10.1016/j.softx.2022.101271)
[![Version](https://img.shields.io/pypi/v/pyfdm)](https://pypi.org/project/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
| Cirteria identification                          | [[6]](#ref6) |
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
| Borda                                        | [[3]](#ref3) |
| Improved Borda                               | [[4]](#ref4) |
| Dominance directed graph                     | [[2]](#ref2) |
| Half-quadratic compromise                    | [[5]](#ref5) |
| ICRA - Iterative Compromise Ranking Analysis | [[1]](#ref1) |
| Rank position method                         | [[3]](#ref3) |

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

<a name="ref1">**[1]**</a> Paradowski, B., Kizielewicz, B., Shekhovtsov, A., & Sałabun, W. (2022, September). The Iterative Compromise Ranking Analysis (ICRA)-The New Approach to Make Reliable Decisions. In Special Sessions in the Advances in Information Systems and Technologies Track of the Conference on Computer Science and Intelligence Systems (pp. 151-170). Cham: Springer Nature Switzerland.
<a name="ref2">**[2]**</a> Xiao, J., Xu, Z., & Wang, X. (2023). An improved MULTIMOORA with CRITIC weights based on new equivalent transformation functions of nested probabilistic linguistic term sets. Soft Computing, 1-18.
<a name="ref3">**[3]**</a> Altuntas, S., Dereli, T., & Yilmaz, M. K. (2015). Evaluation of excavator technologies: application of data fusion based MULTIMOORA methods. Journal of Civil Engineering and Management, 21(8), 977-997.
<a name="ref4">**[4]**</a> Wu, X., Liao, H., Xu, Z., Hafezalkotob, A., & Herrera, F. (2018). Probabilistic linguistic MULTIMOORA: A multicriteria decision making method based on the probabilistic linguistic expectation function and the improved Borda rule. IEEE transactions on Fuzzy Systems, 26(6), 3688-3702.
<a name="ref5">**[5]**</a> Mohammadi, M., & Rezaei, J. (2020). Ensemble ranking: Aggregation of rankings produced by different multi-criteria decision-making methods. Omega, 96, 102254.
<a name="ref6">**[6]**</a> Kizielewicz, B., Wątróbski, J., & Sałabun, W. (2020). Identification of relevant criteria set in the MCDA process—Wind farm location case study. Energies, 13(24), 6548.
