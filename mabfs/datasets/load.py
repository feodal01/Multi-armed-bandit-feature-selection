import os
from typing import Tuple

import arff
import pandas as pd

from datasets.settings import numeric_features_internet_usage, \
    categorical_features_internet_usage

current_path = os.path.dirname(os.path.realpath(__file__))


def internet_usage() -> Tuple[list, list, pd.DataFrame, pd.Series]:
    path = f'{current_path}/internet_usage.arff'
    data_all = arff.load(open(path, 'r'))
    columns = [x[0] for x in data_all['attributes']]
    data = pd.DataFrame(data_all['data'])
    data.columns = columns
    x_columns = [col for col in data.columns if col != 'Actual_Time']
    x = data[x_columns]
    x.loc[:, 'Age'] = x.loc[:, 'Age'].replace('Not_Say', '-1')
    x[numeric_features_internet_usage] = \
        x[numeric_features_internet_usage].astype(int)

    return categorical_features_internet_usage, \
        numeric_features_internet_usage, x,  data['Actual_Time']


def madelon() -> Tuple[list, list, pd.DataFrame, pd.Series]:
    path = f'{current_path}/phpfLuQE4.arff'
    data_all = arff.load(open(path, 'r'))
    columns = [x[0] for x in data_all['attributes']]
    x_columns = [col for col in columns if col != 'Class']
    data = pd.DataFrame(data_all['data'])
    data.columns = columns
    x = data[x_columns]
    numeric_features = x_columns
    categorical_features = list()

    return categorical_features, numeric_features, x, data['Class']
