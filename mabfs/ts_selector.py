import itertools
import joblib
import yaml
import random
from pathlib import Path
from typing import Dict

from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from multiprocessing import Pool
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import mutual_info_score

from distributions import BetaDistribution


class ThompsonSamplingFeatureSelection:
    def __init__(self, 
                 model, 
                 scoring, 
                 X, 
                 y,
                 groups=None,
                 result_folder: str = '',
                 optimization_steps: int = 100,
                 exploration_coef: float = 0.5,
                 desired_number_of_features: int = 10,
                 cv_splits: int = 3,
                 is_regression: bool = True,
                 n_jobs: int = 1):
        
        
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
        
        self.X = X
        self.y = y
        self.groups = groups
        self.splits = cv_splits
        self.exploration_coef = exploration_coef
        self.desired_number_of_features = desired_number_of_features
        self.features = list(self.X.columns)
        
        self.mutual_information = None
        self.iredundancy_matrix = None
        self._this_relevance = None
        self._this_redundancy = None

    def init_distributions_beta(self):
        # init distributions for every feature
        for feat in self.features:
            if self.feat_distributions is None:
                self.feat_distributions = {}
            self.feat_distributions[feat] = BetaDistribution(a=1, b=1)

    def generative_oracle_beta(self):
        # выбираем фичи, которые будем пробовать на каждом шаге
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

    def calculate_mutual_info_regression(self) -> Dict[str, float]:
        result = {}
        columns = self.X.columns
        with Pool(processes=self.n_jobs) as p:
            for col in columns:
                result[col] = p.apply(mutual_info_regression, args=(self.X[col].values.reshape(-1, 1), self.y.values))[0]
        self.mutual_information = result
    
    def calculate_mutual_info_classification(self) -> Dict[str, float]:
        result = {}
        columns = self.X.columns
        with Pool(processes=self.n_jobs) as p:
            for col in columns:
                result[col] = p.apply(mutual_info_classif, args=(self.X[col].values.reshape(-1, 1), self.y.values))[0]
        self.mutual_information = result
        
    def calculate_mutual_redundancy(self) -> Dict[str, float]:
        result = {}
        columns = self.X.columns
        with Pool() as p:
            for i in range(len(columns)):
                for j in range(i+1, len(columns)):
                    col1 = columns[i]
                    col2 = columns[j]
                    key = f"{col1}_{col2}"
                    result[key] = p.apply(mutual_info_regression, args=(self.X[col1].values.reshape(-1, 1), 
                                                                        self.X[col2]))
        self.iredundancy_matrix = result
        
    def calculate_information_relevance(self):
        return sum([self.mutual_information[idn] for idn in self.current_features]) / len(self.current_features)
    
    def calculate_information_redundancy(self):
        all_coefs = [self.iredundancy_matrix.get(f'{f[0]}_{f[1]}', None) for f in
                     itertools.combinations(self.current_features, r=2)]
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
        
    def update_best(self, metric, score, step):
        self.current_best_metric = metric
        self.current_best_score = score
        self.best_features = self.current_features
        for feat in self.current_features:
            self.feat_distributions[feat].update(1)
            
        # print(f'step {step} new best features:', self.best_features)
        print(f'metric: {round(metric, 4)} score: {round(score, 4)}')
        
        with open(Path(self.result_folder) / 'feature_importances.yaml', 
                  'w', encoding='utf-8') as iof:
            yaml.dump(self.best_features,
                      stream=iof,
                      default_flow_style=False,
                      sort_keys=False, allow_unicode=True)

    def one_step(self, step: int):
        self.generative_oracle_beta()
        metric = self.calculate_metric()
        self._this_relevance = self.calculate_information_relevance()  # x & y
        self._this_redundancy = self.calculate_information_redundancy()  # x & x
        if self.is_regression:
            score = metric - (self._this_relevance - self._this_redundancy)
        else:
            score = metric + (self._this_relevance - self._this_redundancy)
        
        if self.current_best_metric is None:
            self.update_best(metric, score, step)
            
        elif self.is_regression and (self.current_best_score > score):
            self.update_best(metric, score, step)
                
        elif not self.is_regression and (self.current_best_score < score):
            self.update_best(metric, score, step)
