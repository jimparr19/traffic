import os
import pandas as pd

from traffic.model import EnhancedModel

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
TRAIN_DATA_DIR = os.path.join(os.path.dirname(__file__), '../src/traffic/data')

# generate model
nodes_df = pd.read_csv(os.path.join(TRAIN_DATA_DIR, 'road_network_nodes.csv'))
edges_df = pd.read_csv(os.path.join(TRAIN_DATA_DIR, 'road_network_edges.csv'))
trips_df = pd.read_csv(os.path.join(TRAIN_DATA_DIR, 'trips.csv'))

enhanced_model = EnhancedModel(node_data=nodes_df, edge_data=edges_df)
enhanced_model.add_trips(trip_data=trips_df)

# test model
test_df = pd.read_csv(os.path.join(TEST_DATA_DIR, 'road_network_edges_ground_truth_test.csv'))

test_df['predicted_travel_time_avg'] = [enhanced_model.network.edges[(x.u_id, x.v_id)]['duration_mean'] for x in
                                        test_df.itertuples()]
test_df['predicted_travel_time_std'] = [enhanced_model.network.edges[(x.u_id, x.v_id)]['duration_std'] for x in
                                        test_df.itertuples()]

test_df['travel_time_avg_err'] = test_df['travel_time_avg'] = test_df['predicted_travel_time_avg']

print(test_df.head())
print(test_df['travel_time_avg_err'].abs().mean())
