import pytest
import numpy as np
from assg.tasks import task_2_1_load_data
from assg.tasks import pca


@pytest.fixture
def data():
    X = task_2_1_load_data()
    return X

def test_assg_data(data):
    X = data

    U, S, mu, sigma = pca(X)
    
    # test U principal components
    assert U.shape == (2,2)
    expected_U = np.array(
        [[-0.70710678, -0.70710678],
            [-0.70710678,  0.70710678]]            
    )
    assert np.all(np.isclose(U, expected_U))

    # test S explained variances vectors
    assert S.shape == (2,)
    expected_S = np.array([1.73553038, 0.26446962])
    assert np.all(np.isclose(S, expected_S))

    # test expected mu and sigma were found from original data
    expected_mu = np.array([3.98926528, 5.00280585])
    assert np.all(np.isclose(mu, expected_mu))

    expected_sigma = np.array([1.16126017, 1.01312201])
    assert np.all(np.isclose(sigma, expected_sigma))

