from pathlib import Path
import yaml
import random

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import scipy.stats as stats
from sklearn.feature_selection import mutual_info_classif

from distributions import BetaDistribution


class ThompsonSamplingFeatureSelection:
    def __init__(self, model, scoring, X, y, 
                 groups=None,
                 result_folder='',
                 steps_to_wait=100, 
                 exploration_coef=0.5,
                 desired_number_of_features=10,
                 splits=3):
        
        self.model = model
        self.scoring = scoring
        self.steps = steps_to_wait
        self.current_best_metric = None
        self.best_features = None
        self.feat_distributions = None
        self.X = X
        self.y = y
        self.groups = groups
        self.splits = splits
        self.exploration_coef = 0.5
        self.desired_number_of_features = desired_number_of_features
        self.features = list(self.X.columns)
        self.init_distributions()
        self.current_features = []
        self.previous_score = None
        self.previous_features = []
        self.result_folder = Path(result_folder)
        
    def init_distributions(self):
        for feat in self.features:
            if self.feat_distributions is None:
                self.feat_distributions = {}
                
            self.feat_distributions[feat] = BetaDistribution(a=0.5, b=0.5)
                
    def generative_oracle(self):
        et = np.random.uniform()
        if self.exploration_coef > et:
            self.current_features = random.sample(self.features, self.desired_number_of_features)
        else:
            distr = {}
            for feat in self.features:
                distr[feat] = self.feat_distributions[feat].sample()
            self.current_features = [k for k,v in distr.items() if v>=min(sorted(distr.values())[-t.desired_number_of_features:])]
            
    def calculate_metric(self):
        metric = cross_val_score(estimator=self.model, 
                                 X=self.X.loc[:, self.current_features], 
                                 y=self.y, 
                                 groups=self.groups,
                                 cv=self.splits,
                                 scoring = self.scoring)
            
        return sum(metric) / len(metric)
    
    def calc_mutual_info_regression(self, x, y):
        return mutual_info_regression(x.reshape(-1,1), y)
        
    def calculate_information_redundancy(self):
        tmp_x = self.X.loc[:, self.current_features].copy(deep=True)
        vf = np.vectorize(self.calc_mutual_info_regression, signature='(n),(n)->()')
        result = pd.DataFrame(vf(tmp_x.T.values, tmp_x.T.values[:, None]))
        iredundancy = result.mask(np.triu(np.ones(result.shape, dtype=np.bool_))).sum().sum() / len(self.current_features)
        
        return iredundancy
        
    def calculate_information_relevance(self):
        tmp_x = self.X.loc[:, self.current_features].copy(deep=True)
        tmp_x = tmp_x.replace([np.inf, -np.inf], np.nan,).fillna(0)
        irelevance = sum(mutual_info_classif(tmp_x, self.y)) / len(self.current_features)
        
        return irelevance
            
    def select_best_features(self):
        
        self.init_distributions()
        
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
            
            if (self.current_best_metric is None) or (metric > self.current_best_metric):
                # metric is better so we want to update best anyway
                self.current_best_metric = metric
                self.best_features = self.current_features
                print(f'step {step} new best features:', self.best_features)
                print(f'm {round(metric,4)} bm {round(self.current_best_metric, 4)}')
                
                with open(self.result_folder / 'feature_importances.yaml', 'w', encoding='utf-8') as iof:
                    yaml.dump(self.best_features,
                              stream=iof,
                              default_flow_style=False,
                              sort_keys=False, allow_unicode=True)

            if (self.previous_score is None) or (score > self.previous_score):
                update_distr = True

            # Update distributions
            #######################################################
            # features_to_update = list(set(self.current_features).intersection(self.previous_features))
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
