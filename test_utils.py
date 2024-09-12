import numpy as np
import pytest
from utils import *

def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = dot_product(vector1, vector2)
    
    assert result == 32, f"Expected 32, but got {result}"
    
def test_cosine_similarity():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = cosine_similarity(vector1, vector2)
    
    # Calculating the expected cosine similarity manually
    dot_prod = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    expected_result = dot_prod / (norm1 * norm2)
    
    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_nearest_neighbor():
    vectors = np.array([[1, 2], [2, 3], [3, 4]])
    query_vector = np.array([2.1, 3.1])
    
    result = nearest_neighbor(query_vector, vectors)
    
    # Calculate the distances manually
    distances = np.linalg.norm(vectors - query_vector, axis=1)
    expected_index = np.argmin(distances)
    
    assert result == expected_index, f"Expected index {expected_index}, but got {result}"
