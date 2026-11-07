"""Geodesic icosphere meshes represented as edge-flow graphs."""

import itertools

import numpy as np

from aerotvc.graph import Graph


def _icosahedron():
    golden_ratio = (1 + np.sqrt(5)) / 2
    points = []
    for first, second in itertools.product((-1, 1), repeat=2):
        points.extend((
            (0, first, second * golden_ratio),
            (first, second * golden_ratio, 0),
            (second * golden_ratio, 0, first),
        ))
    points = np.asarray(points, dtype=float)
    points /= np.linalg.norm(points, axis=1)[:, None]

    edges = {
        (left, right)
        for left in range(len(points))
        for right in range(left + 1, len(points))
        if np.isclose(np.linalg.norm(points[left] - points[right]),
                      2 / np.sqrt(1 + golden_ratio**2))
    }
    faces = [
        (first, second, third)
        for first in range(len(points))
        for second in range(first + 1, len(points))
        for third in range(second + 1, len(points))
        if ((first, second) in edges
            and (first, third) in edges
            and (second, third) in edges)
    ]
    if len(faces) != 20:
        raise RuntimeError("failed to construct the icosahedron faces")
    return points, faces


def icosphere(frequency=1):
    """Return a unit geodesic icosphere subdivided at ``frequency``."""
    if (not isinstance(frequency, (int, np.integer))
            or isinstance(frequency, (bool, np.bool_))
            or frequency < 1):
        raise ValueError("frequency must be a positive integer")
    frequency = int(frequency)

    base_points, faces = _icosahedron()
    points = list(base_points)
    edge_nodes = {}
    face_nodes = {}
    triangles = []

    def node_index(face, weights):
        nonzero = [index for index, weight in enumerate(weights) if weight]
        if len(nonzero) == 1:
            return face[nonzero[0]]

        if len(nonzero) == 2:
            endpoints = [face[index] for index in nonzero]
            low, high = sorted(endpoints)
            high_weight = weights[face.index(high)]
            key = (low, high, high_weight)
            if key not in edge_nodes:
                low_weight = frequency - high_weight
                point = (low_weight * base_points[low]
                         + high_weight * base_points[high]) / frequency
                point /= np.linalg.norm(point)
                edge_nodes[key] = len(points)
                points.append(point)
            return edge_nodes[key]

        key = (face, weights[1], weights[2])
        if key not in face_nodes:
            point = sum(
                weight * base_points[vertex]
                for vertex, weight in zip(face, weights)
            ) / frequency
            point /= np.linalg.norm(point)
            face_nodes[key] = len(points)
            points.append(point)
        return face_nodes[key]

    for face in faces:
        grid = {}
        for first in range(frequency + 1):
            for second in range(frequency - first + 1):
                weights = (frequency - first - second, first, second)
                grid[(first, second)] = node_index(face, weights)

        for first in range(frequency):
            for second in range(frequency - first):
                lower_left = grid[(first, second)]
                lower_right = grid[(first + 1, second)]
                upper_left = grid[(first, second + 1)]
                triangles.append((lower_left, lower_right, upper_left))
                if first + second + 2 <= frequency:
                    upper_right = grid[(first + 1, second + 1)]
                    triangles.append((lower_right, upper_right, upper_left))

    edges = set()
    for first, second, third in triangles:
        for left, right in ((first, second), (second, third), (third, first)):
            edges.add((min(left, right), max(left, right)))

    expected_triangles = 20 * frequency**2
    expected_nodes = 10 * frequency**2 + 2
    expected_edges = 30 * frequency**2
    if (len(triangles) != expected_triangles
            or len(points) != expected_nodes
            or len(edges) != expected_edges):
        raise RuntimeError("icosphere subdivision produced invalid topology")

    tails, heads = zip(*sorted(edges))
    return Graph(len(points), tails, heads)
