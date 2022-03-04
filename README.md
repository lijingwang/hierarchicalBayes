## Hierarchical Bayesian inversion of global variables and large-scale spatial fields
An open-source Python package on the hierarchical Bayesian framework.

Highlight: 
- Hierarchical Bayesian framework for both global and spatial variables. 
- Machine Learning-based inversion
- Local embedding: local principal component analysis

Author: Lijing Wang<sup>1</sup>, Peter Kitanidis<sup>2</sup>, Jef Caers<sup>1</sup>

<sup>1</sup> Department of Geological Sciences, Stanford University

<sup>2</sup> Department of Civil and Environmental Engineering, Stanford University

This paper is under review at Water Resources Research. [Preprint link](https://www.essoar.org/doi/abs/10.1002/essoar.10508754.1)

### Case study of this hierarchical Bayesian framework

- Case 1, linear forward modeling, volume averaging: [case1.ipynb](https://github.com/lijingwang/hierarchicalBayes/blob/master/case1.ipynb)

- Case 2, non-linear forward modeling, pumping test: [case2.ipynb](https://github.com/lijingwang/hierarchicalBayes/blob/master/case2.ipynb)

- Case 3, non-linear forward modeling, 3D floodplain system: [case3.ipynb](https://github.com/lijingwang/hierarchicalBayes/blob/master/case3.ipynb)

Example datasets are too large to be hosted on GitHub. You can download our example datasets from this Google Drive link. 
Or you can generate datasets using MC_case*.py in /utils.

### Methods in /inversion_methods
- Nonlinear global variables inversion using ML-based method: nonlinear_inverse_theta_jointML.py
- Nonlinear spatial variables inversion using ES-MDA: nonlinear_inverse_m_ES.py
- Local PCA: local_inversion_localPCA.py
