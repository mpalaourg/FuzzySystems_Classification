# Fuzzy Systems 4th Assignment, AUTh [2019]
> Create a fuzzy TSK model to classify samples of multivariable, nonlinear functions.

> Dataset used	-> [Isolet Dataset](https://archive.ics.uci.edu/ml/datasets/isolet)

> MATLAB Toolbox needed -> Fuzzy Logic Toolbox

<p align="justify">
This is an <i>experimental</i> application developed as part of the course "Fuzzy Systems" assignment, that took place in the Department of Electrical & Computer Engineering at Aristotle University of Thessaloniki in 2019.
</p>

<p align="justify">
The goal is to find the best model parameters to clasify the samples of the Isolet dataset. Because of the dimensionality of the dataset, the techniques used at Task 1 can't be applied here. So, at first i select the "main" features. Then, through the use of clustering methods to separate the Input space <i>{Algorithm used -> Fuzzy C-means}</i> and the initialization of the fuzzy sets on the resulting groups/clusters, will result that the number of rules depends on the number of clusters and not on the number of inputs. Finally, to find the best pair for NumberOfRules [NR] and NumberOfFeatures [NF], a parametric analysis (GridSearch) was performed, based on a 5-fold Cross-Validation.

For testing <b>NR = [4 8 12 16 20]</b> && <b>NF = [3 9 16 21]</b>.

After the end of GridSearch, the Final_Model (Best Model) was furthermore trained to visualize some results.
</p>
---

## Status

As of the completion of the project, it will NOT be maintained. By no means should it ever be considered stable or safe to use, as it may contain incomplete parts, critical bugs and security vulnerabilities.

---

## Support

Reach out to me:

- [mpalaourg's email](mailto:gbalaouras@gmail.com "gbalaouras@gmail.com")

---

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mpalaourg/FuzzySystems_Classification/blob/master/LICENSE)
