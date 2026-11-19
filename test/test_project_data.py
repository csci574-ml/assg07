import pytest
import numpy as np
from assg.tasks import task_2_1_load_data
from assg.tasks import feature_normalize
from assg.tasks import project_data
from assg.tasks import pca


@pytest.fixture
def data():
    X = task_2_1_load_data()
    return X

def test_assg_data(data):
    X = data 

    X_norm, mu, sigma = feature_normalize(X)
    U, S, mu, sigma = pca(X)
    K = 1
    Z = project_data(X_norm, U, K)

    assert Z.shape == (50,1)
    expected_Z = np.array(
        [[ 1.49631261],
        [-0.92218067],
        [ 1.22439232],
        [ 1.64386173],
        [ 1.2732206 ],
        [-0.97681976],
        [ 1.26881187],
        [-2.34148278],
        [-0.02999141],
        [-0.78171789],
        [-0.6316777 ],
        [-0.55280135],
        [-0.0896816 ],
        [-0.5258541 ],
        [ 1.56415455],
        [-1.91610366],
        [-0.88679735],
        [ 0.95607375],
        [-2.32995679],
        [-0.47793862],
        [-2.21747195],
        [ 0.38900633],
        [-1.78482346],
        [ 0.05175486],
        [ 1.66512392],
        [ 0.50813572],
        [-1.23711018],
        [-1.17198677],
        [ 0.84221686],
        [-0.00693174],
        [-0.22794195],
        [-1.51309518],
        [ 1.33874082],
        [-0.5925244 ],
        [ 0.67907605],
        [-1.35298   ],
        [ 1.68749495],
        [-1.39235931],
        [ 2.55992598],
        [-0.27850702],
        [-0.97677692],
        [ 0.88820006],
        [ 1.29666127],
        [-0.98966774],
        [ 1.81272352],
        [-0.27196356],
        [ 3.19297722],
        [ 1.21299151],
        [ 0.36792871],
        [-1.44264131]]        )
    assert np.all(np.isclose(Z, expected_Z))

