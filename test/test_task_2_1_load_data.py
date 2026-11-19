import pytest
import numpy as np
from assg.tasks import task_2_1_load_data


@pytest.fixture
def data():
    X = task_2_1_load_data()
    return X

def test_loaded_types(data):
    X = data
    assert isinstance(X, np.ndarray)

def test_X_properties(data):
    X = data
    assert X.shape == (50, 2)
