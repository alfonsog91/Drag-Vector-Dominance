import itertools

import numpy as np
import pytest


@pytest.fixture
def icosahedron():
    from aerotvc.graph import Graph

    golden = (1 + np.sqrt(5)) / 2
    vertices = []
    for u, v in itertools.product((-1, 1), repeat=2):
        vertices.extend(((0, u, v * golden), (u, v * golden, 0),
                         (v * golden, 0, u)))
    vertices = np.array(vertices)
    edges = [(i, j) for i in range(12) for j in range(i + 1, 12)
             if np.isclose(np.linalg.norm(vertices[i] - vertices[j]), 2)]
    assert len(edges) == 30
    tails, heads = zip(*edges)
    return Graph(12, tails, heads)
