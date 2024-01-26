# Pg 25: Using plotly instead. Opens in browser tab.

from common import *

import plotly.express as px


def main(model_type: ModelType) -> None:
    # Download and prepare the data.
    life_sat, x, y = get_data()

    # Create an interactive plot using Plotly
    fig = px.scatter(life_sat,
                     x='GDP per capita (USD)',
                     y='Life satisfaction',
                     hover_name='Country',  # Country names will be shown on hover
                     title='Life Satisfaction vs GDP per Capita')
    fig.show()


if __name__ == '__main__':
    main(ModelType.KNearestNeighbor)
