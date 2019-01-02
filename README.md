<p align="center">
  <a href="https://badge.fury.io/py/pysf"><img src="https://badge.fury.io/py/pysf.svg" alt="pysf version" height="18"></a>
  <a href="https://opensource.org/licenses/BSD-3-Clause"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg" alt="License"></a>
</p>

![pysf](https://github.com/alan-turing-institute/pysf/raw/master/docs/_static/logo.png)

Supervised forecasting of sequential data in Python.

## Intro

_Supervised forecasting_ is the machine learning task of making predictions for sequential data like time series (_forecasting_) by exploiting independent examples of the same underlying relationship (_supervised learning_). Learning is flexible enough to incorporate metadata as well as sequential data.

## Package features

* Store and safely manipulate multi-series data and its metadata in a custom data container.
* Define your own machine learning prediction strategies to operate on this data. Make use of tuning and pipelining objects to build composite prediction strategies. Use the widely-adopted fit/predict workflow throughout.
* Plug in classical forecasting or supervised learning-based predictors into a framework that adapts them to the supervised forecasting task. Interface with popular machine learning & forecasting frameworks, such as [scikit-learn](https://scikit-learn.org/stable/), [keras](https://keras.io/) and [statsmodels](https://www.statsmodels.org/stable/index.html). 
* Empirically estimate multiple predictors' generalisation performance using nested resampling schemes, in a statistically sound manner. Compare predictors to baselines.


## Getting started

### Documentation

* Have a look at the [demonstration Jupyter notebook](examples/Walkthrough.ipynb) for a tutorial.
* [API documentation](https://alan-turing-institute.github.io/pysf) is hosted on GitHub Pages.

### Installation

You can install pysf using the [pip](https://pypi.org/project/pysf/) package management system. If you have pip installed, simply run
```
pip install pysf
```
to install the latest release of pysf.

In addition to the package, you will need the following prerequisites to take advantage of pysf's full functionality.

### Prerequisites:

* [pandas](https://pandas.pydata.org/pandas-docs/stable/install.html) 0.20 or higher
* [keras](https://keras.io/#installation) 2.0 or higher
* [scikit-learn](https://scikit-learn.org/stable/install.html)
* [xarray](http://xarray.pydata.org/en/stable/installing.html)
* [scipy](https://scipy.org/install.html)
* [numpy](https://scipy.org/install.html)
* [matplotlib](https://matplotlib.org/users/installing.html)

These are also required, but should be part of your Python distribution:
* abc
* logging

To use keras for deep learning:
* Make sure you [install](https://keras.io/#installation) keras and at least one backend engine. pysf has been tested against TensorFlow and Theano as backends. 
* If using TensorFlow as a backend, you will typically need to install [dask](http://docs.dask.org/en/latest/install.html) 0.15 or higher.

## Contributions

### How to cite

Coming soon!

### How to contribute

We welcome contributions! 

* You can suggest new features or report bugs by creating a [new issue](https://github.com/alan-turing-institute/pysf/issues/new). Please check the [list of issues](https://github.com/alan-turing-institute/pysf/issues) first.
* If you have made a change for an open issue, please submit a pull request linking to that issue.
* If you would like to improve the documentation, please go right ahead and submit a pull request.

### Contributors

* Ahmed Guecioueur ([@ahmedgc](https://github.com/ahmedgc)) is the original author of this package.

### Copyright and license

Code and documentation copyright 2018 [Ahmed Guecioueur](https://github.com/ahmedgc). Code released under the BSD-3-Clause License. 