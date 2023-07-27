import itertools
import random
from multiprocessing import Pool, Manager, Process
from numbers import Integral
# from multiprocess import Pool, Manager, Process
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import MetaEstimatorMixin, BaseEstimator, clone, _fit_context
from sklearn.feature_selection import mutual_info_regression, \
    mutual_info_classif, SelectorMixin
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_is_fitted

import distributions


class ThompsonSamplingFeatureSelection(SelectorMixin,
                                       MetaEstimatorMixin,
                                       BaseEstimator,
                                       ):

    def __init__(self,
                 estimator,
                 *,
                 scoring,
                 result_folder: str = None,
                 optimization_steps: int = 100,
                 exploration_coef: float = 0.5,
                 n_features_to_select: int = 10,
                 cv: int = 3,
                 is_regression: bool = True,
                 n_jobs: int = 1,
                 iredundancy_matrix: dict = None,
                 verbose: int = 1
                 ):

        '''

        :type iredundancy_matrix: dict - precalculated iredundancy_matrix
        :param estimator: model will be used to select features.
        Must have sklearn interface
        :param scoring: use sklearn's make_scorer for target metric
        :param result_folder: where to store yaml file with best features
        :param optimization_steps: how many optimization steps
        :param exploration_coef: at exploration step features will
        be selected totally randomly.
        This is share of exporation steps in all steps
        :param n_features_to_select: how many features will be selected
        :param cv: number of splits for cross validation
        :param is_regression: True if regression task, False if classification
        :param n_jobs: number of jobs
        '''

        self.estimator = estimator
        self.estimator_ = clone(self.estimator)
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.steps = optimization_steps
        self.is_regression = is_regression
        self.result_folder = result_folder
        self.verbose = verbose
        self.cv = cv
        self._exploration_coef = exploration_coef
        self.n_features_to_select = n_features_to_select
        self._iredundancy_matrix = iredundancy_matrix

        self._best_metric = None
        self._best_score = None
        self._best_features = None
        self._feat_distributions = None
        self._current_features = None
        self._n_features_to_select = None
        self._features = None
        self._features_history = list()
        self._mutual_information = None
        self._this_relevance = None
        self._this_redundancy = None
        self._result_dict = None
        self.support_ = None

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_

    @_fit_context(
        prefer_skip_nested_validation=False
    )
    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            groups: pd.Series = None,
            **fit_params
            ):
        return self._fit(X, y, groups, **fit_params)

    def _fit(self,
             X: pd.DataFrame,
             y: pd.Series,
             groups: pd.Series = None, **fit_params):

        tags = self._get_tags()
        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csc",
            ensure_min_features=2,
            force_all_finite=not tags.get("allow_nan", True),
            multi_output=True,
        )

        # Initialization
        n_features = X.shape[1]

        self._features = list(X.columns)
        if self.n_features_to_select is None:
            self._n_features_to_select = n_features // 2
        elif isinstance(self.n_features_to_select, Integral):  # int
            self._n_features_to_select = self.n_features_to_select
        else:  # float
            self._n_features_to_select = int(n_features * self.n_features_to_select)

        self._init_distributions_beta()
        if self.verbose > 0:
            print('Calculating mutual iformation ...')
        if self.is_regression:
            self._calculate_mutual_info_regression(X, y)
        else:
            self._calculate_mutual_info_classification(X, y)

        if self.verbose > 0:
            print('Calculating iformation redundancy ...')

        self._calculate_mutual_redundancy(X)

        for step in range(self.steps):
            self._one_step(step, X, y, groups, **fit_params)

        # Set final attributes
        self.estimator_.fit(X[:, self._best_features], y, **fit_params)
        self._support = np.array([True if x in self._best_features
                                  else False for x in self._features])

        return self

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self._best_features

    def get_irredundancy_matrix(self):
        return self._iredundancy_matrix

    def _init_distributions_beta(self) -> None:
        # init distributions for every feature
        for feat in self._features:
            if self._feat_distributions is None:
                self._feat_distributions = {}
            new_distribution = distributions.BetaDistribution(a=1, b=1)
            self._feat_distributions[feat] = new_distribution

    def _generative_oracle_beta(self) -> None:
        # select features for every step
        et = np.random.uniform()
        if self._exploration_coef > et:
            self._current_features = random.sample(self._features,
                                                   self._n_features_to_select)
        else:
            distr = {}
            for feat in self._features:
                distr[feat] = self._feat_distributions[feat].sample()
            self._current_features = [k for k, v in distr.items() if
                                      v >= min(sorted(distr.values())
                                               [-self._n_features_to_select:])]

        if self._current_features in self._features_history:
            self._generative_oracle_beta()
        else:
            self._features_history.append(self._current_features)

    def _cv_scoring(self, X, y, groups, **fit_params) -> float:
        estimator = clone(self.estimator)
        scores = cross_val_score(estimator=estimator,
                                 X=X.loc[:, self._current_features],
                                 y=y,
                                 groups=groups,
                                 cv=self.cv,
                                 scoring=self.scoring,
                                 n_jobs=self.n_jobs,
                                 **fit_params)

        return sum(scores) / len(scores)

    def _calculate_mutual_info_regression(self, X, y) -> None:
        result = {}
        with Pool(processes=self.n_jobs) as p:
            for col in self._features:
                result[col] = p.apply(mutual_info_regression,
                                      args=(X[col].values.reshape(-1, 1),
                                            y.values))[
                    0]
        self._mutual_information = result

    def _calculate_mutual_info_classification(self, X, y) -> None:
        result = {}
        with Pool(processes=self.n_jobs) as p:
            for col in self._features:
                result[col] = p.apply(mutual_info_classif,
                                      args=(X[col].values.reshape(-1, 1),
                                            y.values))[0]
        self._mutual_information = result

    @staticmethod
    def _calc_mutual_redundancy_regression(X,
                                           col_pairs,
                                           result_dict) -> None:
        for col1, col2 in col_pairs:
            mi = mutual_info_regression(X[col1].values.reshape(-1, 1),
                                        X[col2])
            result_dict[(col1, col2)] = mi

    def _calculate_mutual_redundancy(self, X) -> None:

        if self._iredundancy_matrix is not None:
            return None

        manager = Manager()
        result_dict = manager.dict()
        column_pairs = [(self._features[i], self._features[j]) for
                        i in range(len(self._features)) for j in
                        range(i + 1, len(self._features))]
        num_processes = self.n_jobs
        chunk_size = len(column_pairs) // num_processes
        column_pairs_chunks = [column_pairs[i:i + chunk_size] for
                               i in range(0, len(column_pairs), chunk_size)]

        processes = []
        for i in range(num_processes):
            p = Process(target=self._calc_mutual_redundancy_regression,
                        args=(X, column_pairs_chunks[i], result_dict))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        result = {}
        self._result_dict = result_dict
        for (col1, col2), mi in result_dict.items():
            result.update({f"{col1}_{col2}": mi})

        self._iredundancy_matrix = result

    def _calculate_information_relevance(self) -> float:
        return sum([self._mutual_information[idn] for
                    idn in self._current_features]) /\
            len(self._current_features)

    def _calculate_information_redundancy(self):
        all_coefs = [self._iredundancy_matrix.get(f'{f[0]}_{f[1]}', None)
                     for f in
                     itertools.permutations(self._current_features, r=2)]
        all_coefs = [coef for coef in all_coefs if coef is not None]
        iredundancy = sum(all_coefs) / len(all_coefs)
        return iredundancy[0]

    def _update_worst(self) -> None:
        for feat in self._current_features:
            self._feat_distributions[feat].update(0)

    def save_best_features(self) -> None:
        with open(Path(self.result_folder) / 'feature_importances.yaml',
                  'w', encoding='utf-8') as iof:
            yaml.dump(self._best_features,
                      stream=iof,
                      default_flow_style=False,
                      sort_keys=False, allow_unicode=True)

    def _update_best(self, metric, score, step) -> None:
        self._best_metric = metric
        self._best_score = score
        self._best_features = self._current_features
        for feat in self._current_features:
            self._feat_distributions[feat].update(1)

        if self.result_folder is not None:
            self.save_best_features()

        if self.verbose > 0:
            print(f'step: {step} metric: '
                  f'{round(metric, 4)} score: {round(score, 4)}')

    def _one_step(self, step: int, X, y, groups, **fit_params) -> None:
        self._generative_oracle_beta()
        metric = self._cv_scoring(X, y, groups, **fit_params)
        self._this_relevance = self._calculate_information_relevance()  # x&y
        self._this_redundancy = self._calculate_information_redundancy()  # x&x
        if self.is_regression:
            score = metric - (self._this_relevance - self._this_redundancy)
        else:
            score = metric + (self._this_relevance - self._this_redundancy)

        # update distributions with 1
        if self._best_metric is None:
            self._update_best(metric, score, step)

        elif self.is_regression and (self._best_score > score):
            self._update_best(metric, score, step)

        elif not self.is_regression and (self._best_score < score):
            self._update_best(metric, score, step)

        # update distributions with 0
        else:
            self._update_worst()
