import numpy as np
import pandas as pd
import networkx as nx

import scipy.stats as stats

from abc import ABC, abstractmethod

pd.options.mode.chained_assignment = None


class Model(ABC):
    def __init__(self, node_data: pd.DataFrame, edge_data: pd.DataFrame):
        self.network = self.generate_network(edge_data, node_data)
        self.trip_data = None

    @staticmethod
    def generate_network(edge_data: pd.DataFrame, node_data: pd.DataFrame):
        network = nx.Graph()
        for node in node_data.itertuples():
            network.add_node(node.node_id, x=node.x, y=node.y)
        for edge in edge_data.itertuples():
            network.add_edge(edge.u_id, edge.v_id, length=edge.length, duration_mean=None, duration_std=None)
        return network

    def get_trip_nodes(self, origin, destination):
        path_nodes = nx.shortest_path(self.network, source=origin, target=destination, weight='length')
        return path_nodes

    def get_trip_distance(self, origin, destination):
        path_nodes = self.get_trip_nodes(origin, destination)
        return sum([self.network.edges[u, v]['length'] for u, v in zip(path_nodes[:-1], path_nodes[1:])])

    def get_trip_duration_mean(self, origin, destination):
        if self.trip_data is None:
            raise AttributeError("Edge attribute 'duration_mean' not yet computed. Please add trips first.")
        path_nodes = self.get_trip_nodes(origin, destination)
        return sum([self.network.edges[u, v]['duration_mean'] for u, v in zip(path_nodes[:-1], path_nodes[1:])])

    def get_trip_duration_std(self, origin, destination):
        if self.trip_data is None:
            raise AttributeError("Edge attribute 'duration_std' not yet computed. Please add trips first.")
        path_nodes = self.get_trip_nodes(origin, destination)
        return np.sqrt(
            sum([self.network.edges[u, v]['duration_std'] ** 2 for u, v in zip(path_nodes[:-1], path_nodes[1:])]))

    def add_trips(self, trip_data: pd.DataFrame):
        clean_trip_data = trip_data.loc[trip_data.duration > 0]  # remove zero duration
        self.trip_data = clean_trip_data
        self.compute_edge_duration()

    @abstractmethod
    def compute_edge_duration(self):
        pass

    def predict(self, origin, destination):
        if self.trip_data is None:
            raise AttributeError("Class method 'predict' not available. Please add trips first.")
        trip_duration_mean = self.get_trip_duration_mean(origin, destination)
        trip_duration_std = self.get_trip_duration_std(origin, destination)
        return trip_duration_mean, trip_duration_std


class NaiveModel(Model):
    def __init__(self, node_data: pd.DataFrame, edge_data: pd.DataFrame):
        super().__init__(node_data, edge_data)

    def compute_edge_duration(self):
        trip_data = self.trip_data
        trip_data.loc[:, 'distance'] = [self.get_trip_distance(x.origin, x.destination) for x in trip_data.itertuples()]
        trip_data.loc[:, 'velocity'] = trip_data['distance'] / trip_data['duration']
        unit_duration = 1 / trip_data['velocity']
        unit_duration_mean, unit_duration_std = stats.norm.fit(unit_duration)

        for edge in self.network.edges:
            edge_data = self.network.edges[edge]
            edge_data['duration_mean'] = edge_data['length'] * unit_duration_mean
            edge_data['duration_std'] = edge_data['length'] * unit_duration_std


class EnhancedModel(Model):
    def __init__(self, node_data: pd.DataFrame, edge_data: pd.DataFrame, min_samples: int = 100):
        super().__init__(node_data, edge_data)
        self.min_samples = min_samples

    def compute_edge_duration(self):
        trip_data = self.trip_data
        # compute naive distribution based on all trip data
        trip_data.loc[:, 'distance'] = [self.get_trip_distance(x.origin, x.destination) for x in trip_data.itertuples()]
        trip_data.loc[:, 'velocity'] = trip_data['distance'] / trip_data['duration']
        unit_duration = 1 / trip_data['velocity']
        naive_unit_duration_mean, naive_unit_duration_std = stats.norm.fit(unit_duration)

        nx.set_edge_attributes(self.network, {edge: [] for edge in self.network.edges}, 'velocity_samples')
        for trip in trip_data.itertuples():
            path_nodes = self.get_trip_nodes(trip.origin, trip.destination)
            for i in range(1, len(path_nodes)):
                self.network.edges[path_nodes[i - 1], path_nodes[i]]['velocity_samples'].append(trip.velocity)

        for edge in self.network.edges:
            edge_data = self.network.edges[edge]

            if len(edge_data['velocity_samples']) > self.min_samples:
                unit_duration = 1 / np.array(edge_data['velocity_samples'])
                unit_duration_mean, unit_duration_std = stats.norm.fit(unit_duration)
                edge_data['duration_mean'] = edge_data['length'] * unit_duration_mean
                edge_data['duration_std'] = edge_data['length'] * unit_duration_std
            else:
                # if samples are less than min_samples use naive distribution
                edge_data['duration_mean'] = edge_data['length'] * naive_unit_duration_mean
                edge_data['duration_std'] = edge_data['length'] * naive_unit_duration_std


if __name__ == '__main__':
    nodes_df = pd.read_csv('./data/road_network_nodes.csv')
    edges_df = pd.read_csv('./data/road_network_edges.csv')
    trips_df = pd.read_csv('./data/trips.csv')

    naive_model = NaiveModel(node_data=nodes_df, edge_data=edges_df)
    naive_model.add_trips(trip_data=trips_df)

    enhanced_model = EnhancedModel(node_data=nodes_df, edge_data=edges_df, min_samples=1)
    enhanced_model.add_trips(trip_data=trips_df)

    naive_prediction = [(naive_model.predict(x.origin, x.destination), x.duration) for x in
                        naive_model.trip_data.itertuples()]
    naive_prediction_error = np.mean(np.abs([x[0][0] - x[1] for x in naive_prediction]))
    print(naive_prediction_error)

    enhanced_prediction = [(enhanced_model.predict(x.origin, x.destination), x.duration) for x in
                           enhanced_model.trip_data.itertuples()]
    enhanced_prediction_error = np.mean(np.abs([x[0][0] - x[1] for x in enhanced_prediction]))
    print(enhanced_prediction_error)
