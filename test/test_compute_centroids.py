import pytest
import numpy as np
from assg.tasks import task_1_1_load_data
from assg.tasks import find_closest_centroids
from assg.tasks import compute_centroids


@pytest.fixture
def data():
    X = task_1_1_load_data()
    return X

def test_assg_initial_centroids(data):
    X = data

    initial_centroids = np.array([[3.0, 3.0],
                                    [6.0, 2.0],
                                    [8.0, 5.0]])
    K = initial_centroids.shape[0]
    c = find_closest_centroids(X, initial_centroids)
    expected_centroids = np.array(
        [[2.42830111, 3.15792418],
            [5.81350331, 2.63365645],
            [7.11938687, 3.6166844 ]]
    )
    updated_centroids = compute_centroids(X, c, K)
    
    assert updated_centroids.shape == (3, 2)
    assert np.all(np.isclose(updated_centroids, expected_centroids))

def test_two_centroids(data):
    X = data

    # should work when K=2 centroids
    initial_centroids = np.array([[4.0, 1.0], 
                                    [4.0, 6.0]])
    K = initial_centroids.shape[0]
    c = find_closest_centroids(X, initial_centroids)
    expected_centroids = np.array(
        [[4.51321226, 1.95570575],
            [2.05132113, 5.01925539]]
    )
    updated_centroids = compute_centroids(X, c, K)
    
    assert updated_centroids.shape, (2, 2)
    assert np.all(np.isclose(updated_centroids, expected_centroids))

def test_random_dataset():
    # should work with a random dataset and K=5 centroids
    np.random.seed(42)
    X = np.random.rand(20, 20)
    # use first 5 random points as centroids
    initial_centroids = X[:5]
    K = initial_centroids.shape[0]
    c = find_closest_centroids(X, initial_centroids)
    expected_centroids = np.array(
        [[0.55648987, 0.54838734, 0.59442999, 0.57641605, 0.39820266,
            0.51787353, 0.43240153, 0.61809305, 0.48227664, 0.47288888,
            0.30266315, 0.75974241, 0.65150762, 0.27760649, 0.25318302,
            0.33496875, 0.52895405, 0.69230356, 0.65885493, 0.44846882],
            [0.4091162 , 0.44046588, 0.42052743, 0.47177151, 0.66320662,
            0.48666197, 0.53942239, 0.60015434, 0.37273376, 0.14845407,
            0.46906232, 0.1335971 , 0.25188756, 0.77842277, 0.698371  ,
            0.58737261, 0.60684993, 0.24848561, 0.55706702, 0.44796175],
            [0.36852273, 0.74261538, 0.08723627, 0.71382503, 0.56807653,
            0.70164545, 0.50436341, 0.61127605, 0.45310072, 0.23922315,
            0.88947289, 0.79262311, 0.90328563, 0.90403395, 0.55462119,
            0.71169526, 0.44339384, 0.4229734 , 0.37359708, 0.5605615 ],
            [0.29154175, 0.53346401, 0.69707002, 0.59000001, 0.36876511,
            0.72715984, 0.24791295, 0.38871834, 0.28864685, 0.79487154,
            0.81343809, 0.20930706, 0.29654442, 0.60612339, 0.56427789,
            0.61137697, 0.77354028, 0.29659134, 0.68202226, 0.33597763],
            [0.80454045, 0.6064233 , 0.27953981, 0.158827  , 0.3456    ,
            0.25805576, 0.74655738, 0.6085076 , 0.51818817, 0.55123353,
            0.3405792 , 0.36883447, 0.57003872, 0.43790018, 0.5090451 ,
            0.51542348, 0.52307234, 0.34267424, 0.41435288, 0.33553291]]
    )
    updated_centroids = compute_centroids(X, c, K)
    
    assert updated_centroids.shape == (5, 20)
    assert np.all(np.isclose(updated_centroids, expected_centroids))

