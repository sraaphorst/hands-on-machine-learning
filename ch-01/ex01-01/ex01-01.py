# Pg 25: Training and running a linear model using Scikit-Learn
#        for GDP per capita and life satisfaction.

from common import *

import matplotlib.pyplot as plt
import pandas as pd


def main(model_type: ModelType) -> None:
    # Download and prepare the data.
    life_sat, x, y = get_data()

    plt.scatter(life_sat['GDP per capita (USD)'], life_sat['Life satisfaction'])
    # for i, txt in enumerate(life_sat['Country']):
    #     plt.annotate(txt, (life_sat['GDP per capita (USD)'][i], life_sat['Life satisfaction'][i]))
    plt.grid(True)
    plt.xlabel('GDP per capita (USD)')
    plt.ylabel('Life satisfaction')
    plt.axis((23_500, 62_500, 4, 9))
    plt.show()

    # Select a model and train it.
    model = None
    match model_type:
        case ModelType.LinearRegression:
            model = model_type.value()
        case ModelType.KNearestNeighbor:
            model = model_type.value(3)

    model.fit(x, y)

    # Make a new prediction for Cyprus: GDP per capita in 2020 using linear regression.
    # If we had used instance-based learning algorithm, Israel has the closest GDP per capita to Cyrus.
    #    Israel is 7.2, so Cyprus would be predicted to be 7.2.
    # If we did k-NN with k=3, two next-closest countries Lithuania and Slovenia both have 5.9.
    #    Averaging the three values (7.2 + 5.9 + 5.9) / 3 = 6.33.
    x_new = [[37_655.20]]
    prediction = model.predict(x_new)
    print(prediction[0].item())


if __name__ == '__main__':
    main(ModelType.KNearestNeighbor)
