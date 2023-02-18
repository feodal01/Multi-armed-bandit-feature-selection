from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.linear_model import LinearRegression

from .ts_selector import ThompsonSamplingFeatureSelection


def test_selection():
    x, y, coef = make_regression(n_samples=100,
                                 n_features=10,
                                 n_informative=5,
                                 effective_rank=5,
                                 tail_strength=0.7,
                                 noise=0.05,
                                 shuffle=True,
                                 bias=100,
                                 coef=True,
                                 random_state=666)

    x = pd.DataFrame(x)
    y = pd.Series(y)

    # Список значимых фичей
    true_features = np.where(coef > 0)[0]
    model = LinearRegression()
    tsfs = ThompsonSamplingFeatureSelection(model=model,
                                            scoring=make_scorer(mean_absolute_error),
                                            desired_number_of_features=5,
                                            x=x,
                                            y=y,
                                            cv_splits=3,
                                            exploration_coef=0.3,
                                            optimization_steps=1000,
                                            is_regression=True,
                                            n_jobs=36
                                            )
    tsfs.select_best_features()
    assert tsfs.best_features == list(true_features)


if __name__ == "__main__":
    test_selection()
    print("Everything passed")