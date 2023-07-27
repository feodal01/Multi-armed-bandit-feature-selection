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
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from mabfs.ts_selector import ThompsonSamplingFeatureSelection
from datasets.load import madelon

# load madelon dataset
cat, num, x, y = madelon()

# spawn model that will be used for selection
model = RandomForestRegressor(random_state=666, n_jobs=-1)

# spawn tsfs class
tsfs = ThompsonSamplingFeatureSelection(estimator=model,
                                        scoring=make_scorer(mean_absolute_error),
                                        cv=3,
                                        exploration_coef=0.3,
                                        optimization_steps=100_000,
                                        n_features_to_select=20,
                                        is_regression=True,
                                        n_jobs=36,
                                        verbose=1)

# select features
tsfs.fit(X=x, y=y)

# show features
selected_features = tsfs.get_feature_names_out()
```
