import pytest
import numpy as np
from assg.tasks import task_2_1_load_data
from assg.tasks import feature_normalize


@pytest.fixture
def data():
    X = task_2_1_load_data()
    return X

def test_assg_data(data):
    X = data

    X_norm, mu, sigma = feature_normalize(X)

    # test X_norm properties
    assert isinstance(X_norm, np.ndarray)
    assert X_norm.shape, (50,2)
    assert np.isclose(X_norm.mean(), 0.0)
    assert np.isclose(X_norm.std(), 1.0)

    # test expected mu and sigma were found from original data
    expected_mu = np.array([3.98926528, 5.00280585])
    assert np.all(np.isclose(mu, expected_mu))

    expected_sigma = np.array([1.16126017, 1.01312201])
    assert np.all(np.isclose(sigma, expected_sigma))

def test_random_data():
    np.random.seed(42)
    X = np.random.rand(1000, 5)
    X_norm, mu, sigma = feature_normalize(X)

    # test X_norm properties
    assert isinstance(X_norm, np.ndarray)
    assert X_norm.shape == (1000,5)
    assert np.isclose(X_norm.mean(), 0.0)
    assert np.isclose(X_norm.std(), 1.0)

    # test expected mu and sigma were found from original data
    expected_mu = np.array([0.4963718 , 0.50008531, 0.48985751, 0.49780181, 0.50004354])
    assert np.all(np.isclose(mu, expected_mu))

    expected_sigma = np.array([0.2901722 , 0.29694818, 0.28486009, 0.28686249, 0.28891208])
    assert np.all(np.isclose(sigma, expected_sigma))
