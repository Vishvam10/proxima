import numpy as np
import proxima


def test_empty_index():
    index = proxima.Index()

    assert len(index) == 0


def test_create_and_search():
    vectors = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
        ],
        dtype=np.float32,
    )

    index = proxima.Index()

    index.create(vectors)

    assert len(index) == 4
    assert index.dimension() == 2

    query = np.array(
        [0.1, 0.1],
        dtype=np.float32,
    )

    result = index.search(
        query,
        k=2,
    )

    assert len(result) == 2
    assert 0 in result


def test_incremental_add():
    index = proxima.Index()

    index.add(
        np.array(
            [0.0, 0.0],
            dtype=np.float32,
        )
    )

    index.add(
        np.array(
            [1.0, 1.0],
            dtype=np.float32,
        )
    )

    assert len(index) == 2


def test_dimension_validation():
    index = proxima.Index()

    index.add(
        np.array(
            [0.0, 0.0],
            dtype=np.float32,
        )
    )

    try:
        index.add(
            np.array(
                [1.0, 2.0, 3.0],
                dtype=np.float32,
            )
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "expected dimension mismatch"
        )


def test_metrics():
    index = proxima.Index(
        distance_type=proxima.DistanceType.L2
    )

    assert (
        index.distance_type()
        == proxima.DistanceType.L2
    )