from enum import Enum

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


class ModelType(Enum):
    LinearRegression = LinearRegression
    KNearestNeighbor = KNeighborsRegressor


def get_data():
    data_root = 'https://github.com/ageron/data/raw/main'
    life_sat = pd.read_csv(f'{data_root}/lifesat/lifesat.csv')
    x = life_sat[['GDP per capita (USD)']].values
    y = life_sat[['Life satisfaction']].values
    return life_sat, x, y
