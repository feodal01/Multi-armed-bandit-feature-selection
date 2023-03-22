import itertools
import random
# from multiprocessing import Pool, Manager, Process
from multiprocess import Pool, Manager, Process
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.model_selection import cross_val_score

import distributions


class ThompsonSamplingFeatureSelection:
    def __init__(self,
                 model,
                 scoring,
                 x: pd.DataFrame,
                 y: pd.Series,
                 groups: pd.Series = None,
                 result_folder: str = None,
                 optimization_steps: int = 100,
                 exploration_coef: float = 0.5,
                 desired_number_of_features: int = 10,
                 cv_splits: int = 3,
                 is_regression: bool = True,
                 n_jobs: int = 1):

        '''

        :param model: model will be used to select features. Must have sklearn interface
        :param scoring: use sklearn's make_scorer for target metric
        :param x: features
        :param y: target variable
        :param groups: provide here groups for target variable if any
        :param result_folder: where to store yaml file with best features
        :param optimization_steps: how many optimization steps
        :param exploration_coef: at exploration step features will be selected totally randomly.
        This is share of exporation steps in all steps
        :param desired_number_of_features: how many features will be selected
        :param cv_splits: number of splits for cross validation
        :param is_regression: True if regression task, False if classification
        :param n_jobs: number of jobs
        '''

        self.model = model
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.steps = optimization_steps
        self.is_regression = is_regression
        self.result_folder = result_folder

        self.current_best_metric = None
        self.current_best_score = None
        self.best_features = None
        self.feat_distributions = None
        self.current_features = None

        self.X = x
        self.y = y
        self.groups = groups
        self.splits = cv_splits
        self.exploration_coef = exploration_coef
        self.desired_number_of_features = desired_number_of_features
        self.features = list(self.X.columns)
        self.features_history = list()

        self.mutual_information = None
        self.iredundancy_matrix = None
        self._this_relevance = None
        self._this_redundancy = None

    def init_distributions_beta(self):
        # init distributions for every feature
        for feat in self.features:
            if self.feat_distributions is None:
                self.feat_distributions = {}
            self.feat_distributions[feat] = distributions.BetaDistribution(a=1, b=1)

    def generative_oracle_beta(self):
        # select features for every step
        et = np.random.uniform()
        if self.exploration_coef > et:
            self.current_features = random.sample(self.features, self.desired_number_of_features)
        else:
            distr = {}
            for feat in self.features:
                distr[feat] = self.feat_distributions[feat].sample()
            self.current_features = [k for k, v in distr.items() if
                                     v >= min(sorted(distr.values())[-self.desired_number_of_features:])]

        if self.current_features in self.features_history:
            self.generative_oracle_beta()
        else:
            self.features_history.append(self.current_features)

    def calculate_metric(self):
        metric = cross_val_score(estimator=self.model,
                                 X=self.X.loc[:, self.current_features],
                                 y=self.y,
                                 groups=self.groups,
                                 cv=self.splits,
                                 scoring=self.scoring,
                                 n_jobs=self.n_jobs)

        return sum(metric) / len(metric)

    def calculate_mutual_info_regression(self):
        result = {}
        columns = self.X.columns
        with Pool(processes=self.n_jobs) as p:
            for col in columns:
                result[col] = p.apply(mutual_info_regression, args=(self.X[col].values.reshape(-1, 1), self.y.values))[
                    0]
        self.mutual_information = result

    def calculate_mutual_info_classification(self):
        result = {}
        columns = self.X.columns
        with Pool(processes=self.n_jobs) as p:
            for col in columns:
                result[col] = p.apply(mutual_info_classif, args=(self.X[col].values.reshape(-1, 1), self.y.values))[0]
        self.mutual_information = result

    def _calc_mutual_redundancy_regression(self, col_pairs, result_dict):
        for col1, col2 in col_pairs:
            mi = mutual_info_regression(self.X[col1].values.reshape(-1, 1), self.X[col2])
            result_dict[(col1, col2)] = mi

    def calculate_mutual_redundancy(self):
        manager = Manager()
        result_dict = manager.dict()
        col_names = self.X.columns.tolist()
        column_pairs = [(col_names[i], col_names[j]) for i in range(len(col_names)) for j in
                        range(i + 1, len(col_names))]
        num_processes = self.n_jobs
        chunk_size = len(column_pairs) // num_processes
        column_pairs_chunks = [column_pairs[i:i + chunk_size] for i in range(0, len(column_pairs), chunk_size)]

        processes = []
        for i in range(num_processes):
            p = Process(target=self._calc_mutual_redundancy_regression,
                        args=(column_pairs_chunks[i], result_dict))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        result = {}
        self.result_dict = result_dict
        for (col1, col2), mi in result_dict.items():
            result.update({f"{col1}_{col2}": mi})

        self.iredundancy_matrix = result

    def calculate_information_relevance(self):
        return sum([self.mutual_information[idn] for idn in self.current_features]) / len(self.current_features)

    def calculate_information_redundancy(self):
        all_coefs = [self.iredundancy_matrix.get(f'{f[0]}_{f[1]}', None) for f in
                     itertools.permutations(self.current_features, r=2)]
        all_coefs = [coef for coef in all_coefs if coef is not None]
        iredundancy = sum(all_coefs) / len(all_coefs)
        return iredundancy[0]

    def select_best_features(self):
        self.init_distributions_beta()

        print('Calculating mutual iformation ...')
        if self.is_regression:
            self.calculate_mutual_info_regression()
        else:
            self.calculate_mutual_info_classification()
        print('Calculating iformation redundancy ...')
        self.calculate_mutual_redundancy()

        for step in range(self.steps):
            self.one_step(step)

    def update_worst(self):
        for feat in self.current_features:
            self.feat_distributions[feat].update(0)

    def save_best_features(self):
        with open(Path(self.result_folder) / 'feature_importances.yaml',
                  'w', encoding='utf-8') as iof:
            yaml.dump(self.best_features,
                      stream=iof,
                      default_flow_style=False,
                      sort_keys=False, allow_unicode=True)

    def update_best(self, metric, score, step):
        self.current_best_metric = metric
        self.current_best_score = score
        self.best_features = self.current_features
        for feat in self.current_features:
            self.feat_distributions[feat].update(1)

        if self.result_folder is not None:
            self.save_best_features()

        print(f'step: {step} metric: {round(metric, 4)} score: {round(score, 4)}')

    def one_step(self, step: int):
        self.generative_oracle_beta()
        metric = self.calculate_metric()
        self._this_relevance = self.calculate_information_relevance()  # x & y
        self._this_redundancy = self.calculate_information_redundancy()  # x & x
        if self.is_regression:
            score = metric - (self._this_relevance - self._this_redundancy)
        else:
            score = metric + (self._this_relevance - self._this_redundancy)

        # update distributions with 1
        if self.current_best_metric is None:
            self.update_best(metric, score, step)

        elif self.is_regression and (self.current_best_score > score):
            self.update_best(metric, score, step)

        elif not self.is_regression and (self.current_best_score < score):
            self.update_best(metric, score, step)

        # update distributions with 0
        else:
            self.update_worst()
