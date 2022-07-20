import itertools
import multiprocessing
from pathlib import Path

import joblib
import yaml
import random

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from distributions import BetaDistribution


class ThompsonSamplingFeatureSelection:
    def __init__(self, model, scoring, X, y,
                 groups=None,
                 result_folder: str = '',
                 biger_is_better: bool = True,
                 optimization_steps: int = 100,
                 exploration_coef: float = 0.5,
                 desired_number_of_features: int = 10,
                 cv_splits: int = 3,
                 is_regression: bool = False,
                 n_jobs: int = -1):
        
        '''
        :param model: ML model with sklearn api
        :param scoring: scoring function (pls use make_scorer from sklearn)
        :param groups: if data have groups - please specify series with groups. it will be used for cross-validation
        :param result_folder: where to save yaml with best columns
        :param biger_is_better: for metric like accuracy - True, for metric like MAE - False
        :param optimization_steps: how many iterations try to optimize list of fetures
        :param exploration_coef: the probability of selecting random list of features at every optimization step
        :param desired_number_of_features: how many features to select
        :param cv_splits: for cross validation
        :param is_regression: if problem is regression - please specify True
        :param n_jobs: how many cores to use in optimization, if -1 all cores will be used
        '''

        self.model = model
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.steps = optimization_steps
        self.current_best_metric = None
        self.best_features = None
        self.feat_distributions = None
        self.biger_is_better = biger_is_better
        self.X = X
        self.y = y
        self.groups = groups
        self.splits = cv_splits
        self.exploration_coef = exploration_coef
        self.desired_number_of_features = desired_number_of_features
        self.features = list(self.X.columns)
        self.init_distributions()
        self.current_features = []
        self.previous_score = None
        self.previous_features = []
        self.result_folder = Path(result_folder)
        self.mutual_info_x_y = dict()
        self.is_regression = is_regression
        self.iredundancy_matrix = None

    def init_distributions(self):
        for feat in self.features:
            if self.feat_distributions is None:
                self.feat_distributions = {}

            self.feat_distributions[feat] = BetaDistribution(a=1, b=1)

    def generative_oracle(self):
        et = np.random.uniform()
        if self.exploration_coef > et:
            self.current_features = random.sample(self.features, self.desired_number_of_features)
        else:
            distr = {}
            for feat in self.features:
                distr[feat] = self.feat_distributions[feat].sample()
            self.current_features = [k for k, v in distr.items() if
                                     v >= min(sorted(distr.values())[-self.desired_number_of_features:])]

    def calculate_metric(self):
        metric = cross_val_score(estimator=self.model,
                                 X=self.X.loc[:, self.current_features],
                                 y=self.y,
                                 groups=self.groups,
                                 cv=self.splits,
                                 scoring=self.scoring,
                                 n_jobs=self.n_jobs)

        return sum(metric) / len(metric)

    def calc_all_mutual_info(self):

        processes = []
        if self.n_jobs < 0:
            pool = multiprocessing.Pool(joblib.cpu_count())
        else:
            pool = multiprocessing.Pool(self.n_jobs)

        manager = multiprocessing.Manager()
        share_dict = manager.dict()

        for feat_name in self.features:
            processes.append(
                pool.apply_async(self.calc_mutual_info_regression,
                                 args=(share_dict,
                                       feat_name,
                                       self.X[feat_name],
                                       self.y
                                       ),
                                 )
            )

        [p.get() for p in processes]
        self.mutual_info_x_y.update(share_dict)

    def calc_mutual_info_regression(self, share_dict, feat_name, x, y):
        if self.is_regression:
            share_dict[feat_name] = mutual_info_regression(np.array(x).reshape(-1, 1),
                                                           np.array(y).ravel())
        else:
            share_dict[feat_name] = mutual_info_classif(np.array(x).reshape(-1, 1),
                                                        np.array(y).ravel())

    @staticmethod
    def calc_mutual_info_regression_vector(x, y):
        return mutual_info_regression(x.reshape(-1, 1), y)

    def __old_calculate_information_redundancy(self):
        tmp_x = self.X.loc[:, self.current_features].copy(deep=True)
        vf = np.vectorize(self.calc_mutual_info_regression_vector, signature='(n),(n)->()')
        result = pd.DataFrame(vf(tmp_x.T.values, tmp_x.T.values[:, None]))
        iredundancy = result.mask(np.triu(np.ones(result.shape, dtype=np.bool_))).sum().sum() / len(
            self.current_features)

        return iredundancy

    def calculate_information_redundancy(self):
        indexes = [self.features.index(f) for f in self.current_features]
        iredundancy = sum(
            [self.iredundancy_matrix.loc[pair[1], pair[0]] for pair in itertools.combinations(indexes, r=2)]) / len(
            self.current_features)
        return iredundancy

    def calculate_information_relevance(self):
        return sum([self.mutual_info_x_y[idn] for idn in self.current_features]) / len(self.current_features)

    def calculate_information_redundancy_matrix(self):
        vf = np.vectorize(self.calc_mutual_info_regression, signature='(n),(n)->()')
        result = pd.DataFrame(vf(self.X.T.values, self.X.T.values[:, None]))
        iredundancy = result.mask(np.triu(np.ones(result.shape, dtype=np.bool_)))
        self.iredundancy_matrix = iredundancy

    def __old_calculate_information_relevance(self):
        tmp_x = self.X.loc[:, self.current_features].copy(deep=True)
        tmp_x = tmp_x.replace([np.inf, -np.inf], np.nan, ).fillna(0)
        irelevance = sum(mutual_info_classif(tmp_x, self.y)) / len(self.current_features)

        return irelevance

    def select_best_features(self):

        self.init_distributions()
        self.calc_all_mutual_info()
        self.calculate_information_redundancy_matrix()

        for step in range(self.steps):

            if len(self.current_features) > 0:
                self.current_features = []

            self.generative_oracle()

            # Train model
            #######################################################
            metric = self.calculate_metric()
            irelevance = self.calculate_information_relevance()
            iredundancy = self.calculate_information_redundancy()
            score = metric + irelevance + iredundancy

            # Better or not?
            #######################################################
            update_distr = False

            if (self.current_best_metric is None) or \
                    ((metric > self.current_best_metric) and (self.biger_is_better == True)) or \
                    ((metric < self.current_best_metric) and (self.biger_is_better == False)):
                # metric is better so we want to update best anyway
                self.current_best_metric = metric
                self.best_features = self.current_features
                print(f'step {step} new best features:', self.best_features)
                print(f'm {round(metric, 4)} bm {round(self.current_best_metric, 4)}')

                with open(self.result_folder / 'feature_importances.yaml', 'w', encoding='utf-8') as iof:
                    yaml.dump(self.best_features,
                              stream=iof,
                              default_flow_style=False,
                              sort_keys=False, allow_unicode=True)

            if (self.previous_score is None) or \
                    ((score > self.previous_score) and (self.biger_is_better == True)) or \
                    ((score < self.previous_score) and (self.biger_is_better == False)):
                update_distr = True

            # Update distributions
            #######################################################
            features_to_update = self.current_features

            if len(features_to_update) == 0:
                features_to_update = self.current_features

            if update_distr:
                for feat in features_to_update:
                    self.feat_distributions[feat].update(1)
            else:
                for feat in features_to_update:
                    self.feat_distributions[feat].update(0)

            self.previous_score = score
            self.previous_features = self.current_features.copy()
