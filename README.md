# Multi armed bandit feature selection
Feature selector based on Thompson sampling algorithm
Based on: https://epubs.siam.org/doi/pdf/10.1137/1.9781611976700.36

### Descriotion
This package is used to select optimal subset of features to maximize selected metric.
Optimization could be used for both regression and classification.

### How it works
1) calculate information relevance and information redundancy for every feature
2) init beta distribution for every feature
3) sample every beta distribution and select desired number of feature
4) cross validate model
5) calculate resulting score based om CV metric and information relevance and redundancy 
6) update beta distributions

### Usage

```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from mabfs.ts_selector import ThompsonSamplingFeatureSelection

x, y, coef = make_regression(n_samples=1000,
                             n_features=500, 
                             n_informative=10, 
                             effective_rank=5, 
                             tail_strength=0.7,
                             noise=0.05, 
                             shuffle=True, 
                             bias=100,
                             coef=True,
                             random_state=666)

x = pd.DataFrame(x)
y = pd.Series(y)

true_features = np.where(coef > 0)[0]

model = LinearRegression()
tsfs = ThompsonSamplingFeatureSelection(model=model, 
                                        scoring=make_scorer(mean_absolute_error),
                                        desired_number_of_features=10,
                                        X=x, 
                                        y=y, 
                                        cv_splits=3, 
                                        exploration_coef=0.3,
                                        optimization_steps=100000,
                                        is_regression=True,
                                        n_jobs=36
                                       )

tsfs.select_best_features()
```
