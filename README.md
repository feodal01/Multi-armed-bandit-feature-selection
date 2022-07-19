# Multi armed bandit based feature selection
Feature selector based on Thompson sampling algorithm

Based on blueprint: https://epubs.siam.org/doi/pdf/10.1137/1.9781611976700.36

### USAGE

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from ts_selector import ThompsonSamplingFeatureSelection

model = LinearRegression()
x = your_data_here
y = your_data_here

tsfs = ThompsonSamplingFeatureSelection(model=model, 
                                        scoring=make_scorer(mean_absolute_error), 
                                        X=x, y=y, 
                                        is_regression=True,
                                        biger_is_better=False)
                                        
tsfs.select_best_features()
```
