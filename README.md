# traffic

[![Actions Status](https://github.com/jimparr19/traffic/workflows/Python%20package/badge.svg)](https://github.com/jimparr19/traffic/actions)

Details of the approach can be found in [notebooks/0.Exploration.ipynb](./notebooks/0.Exploration.ipynb)

## Installation 

Clone this repo and install the python package and required dependencies using pip

```bash
pip install .
```

Run tests using pytest

```bash
pytest
```

## Example use

```python
import pandas as pd
import numpy as np
from traffic.model import NaiveModel

nodes_df = pd.read_csv('road_network_nodes.csv')
edges_df = pd.read_csv('road_network_edges.csv')
trips_df = pd.read_csv('trips.csv')

naive_model = NaiveModel(node_data=nodes_df, edge_data=edges_df)
naive_model.add_trips(trip_data=trips_df)

naive_prediction = [(naive_model.predict(x.origin, x.destination), x.duration) for x in
                    naive_model.trip_data.itertuples()]
naive_prediction_error = np.mean(np.abs([x[0][0] - x[1] for x in naive_prediction]))
print(naive_prediction_error)
```