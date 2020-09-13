import os
import pytest
import pandas as pd

from traffic.model import NaiveModel, EnhancedModel

pd.options.mode.chained_assignment = None

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture
def node_data():
    node_df = pd.read_csv(os.path.join(DATA_DIR, 'nodes_test_data.csv'))
    return node_df


@pytest.fixture
def edge_data():
    edge_df = pd.read_csv(os.path.join(DATA_DIR, 'edges_test_data.csv'))
    return edge_df


@pytest.fixture
def trip_data():
    trip_df = pd.read_csv(os.path.join(DATA_DIR, 'trips_test_data.csv'))
    return trip_df


@pytest.fixture
def naive_model(node_data, edge_data):
    model = NaiveModel(node_data=node_data, edge_data=edge_data)
    return model


@pytest.fixture
def enhanced_model(node_data, edge_data):
    model = EnhancedModel(node_data=node_data, edge_data=edge_data)
    return model


def test_naive_model_generate_network(node_data, edge_data):
    model = NaiveModel(node_data=node_data, edge_data=edge_data)
    assert model.network


def test_enhanced_model_generate_network(node_data, edge_data):
    model = EnhancedModel(node_data=node_data, edge_data=edge_data)
    assert model.network


def test_naive_model_add_trip_data(naive_model, trip_data):
    naive_model.add_trips(trip_data=trip_data)
    assert naive_model.trip_data is not None


def test_enhanced_model_add_trip_data(enhanced_model, trip_data):
    enhanced_model.add_trips(trip_data=trip_data)
    assert enhanced_model.trip_data is not None


def test_enhanced_model_get_trip_duration_mean(enhanced_model, trip_data):
    origin = trip_data.loc[0, 'origin']
    destination = trip_data.loc[0, 'destination']
    with pytest.raises(AttributeError):
        enhanced_model.get_trip_duration_mean(origin=origin, destination=destination)
    enhanced_model.add_trips(trip_data=trip_data)
    duration_mean = enhanced_model.get_trip_duration_mean(origin=origin, destination=destination)
    assert duration_mean > 0


def test_enhanced_model_get_trip_duration_std(enhanced_model, trip_data):
    origin = trip_data.loc[0, 'origin']
    destination = trip_data.loc[0, 'destination']
    with pytest.raises(AttributeError):
        enhanced_model.get_trip_duration_std(origin=origin, destination=destination)
    enhanced_model.add_trips(trip_data=trip_data)
    duration_std = enhanced_model.get_trip_duration_std(origin=origin, destination=destination)
    assert duration_std > 0


def test_naive_model_predict(naive_model, trip_data):
    origin = trip_data.loc[0, 'origin']
    destination = trip_data.loc[0, 'destination']
    with pytest.raises(AttributeError):
        naive_model.predict(origin=origin, destination=destination)
    naive_model.add_trips(trip_data=trip_data)
    duration_mean, duration_std = naive_model.predict(origin=origin, destination=destination)
    assert duration_std > 0


def test_enhanced_model_predict(enhanced_model, trip_data):
    origin = trip_data.loc[0, 'origin']
    destination = trip_data.loc[0, 'destination']
    with pytest.raises(AttributeError):
        enhanced_model.predict(origin=origin, destination=destination)
    enhanced_model.add_trips(trip_data=trip_data)
    duration_mean, duration_std = enhanced_model.predict(origin=origin, destination=destination)
    assert duration_std > 0


if __name__ == '__main__':
    pytest.main()
