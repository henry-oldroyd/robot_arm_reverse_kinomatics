import abc
import numpy as np
from math import exp
from functools import lru_cache

class Range_Limiter():
    def __init__(self, minimum, maximum):
        self.min, self.max = minimum, maximum

    @lru_cache(16)
    def sigmoid(self, x): 
        return 1/(1+exp(-x))
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def map_scalar_to_target_interval(self, x):
        return (
            (self.max - self.min) 
            * self.sigmoid(x)
        ) + self.min

    def derivative_of_mapping(self, x):
        return (
           (self.max - self.min) 
            * self.sigmoid(x)
            * self.sigmoid_derivative(x) 
        )

class Range_Limit_Vector():
    def __init__(self, intervals):
        self.intervals = tuple(intervals)
        self.vector_size = len(self.intervals)
        self.mappings = tuple((Range_Limiter(*interval) for interval in self.intervals))

    def map_vector_to_target_interval(self, X):
        mapped_values = [
            mapping.map_scalar_to_target_interval(x)
            for x, mapping in zip(X, self.mappings)
        ]
        return np.array(mapped_values)
    
    def derivative_of_mapping(self, X):
        mapping_derivatives = [
            mapping.derivative_of_mapping(x)
            for x, mapping in zip(X, self.mappings)            
        ]
        return np.diag(mapping_derivatives)



class Differential_Operation(abc.ABC):
    @abc.abstractmethod
    def set_parameter(self, value):
        pass

    @abc.abstractmethod
    def function(vector: np.ndarray):
        pass

    @abc.abstractmethod
    def derivative_WR_vector():
        pass

    @abc.abstractmethod
    def derivative_WR_parameter():
        pass

class Translate(Differential_Operation):
    def __init__(self, translation_vector) -> None:
        self._translation_vector = translation_vector
        self._vector_length = len(translation_vector)
    

    def set_parameter(self, value):
        return None

    def function(self, vector: np.ndarray):
        return vector + self._translation_vector
    
    def derivative_WR_vector(self):
        # return identity matrix
        return np.eye(self._vector_length)
    
    def derivative_WR_parameter(self):
        return np.eye(self._vector_length, self._vector_length)
    
class Rotation_About_Vector():
    def __init__(self, rotation_vector) -> None:
        self._rotation_vector = rotation_vector
        self._parameter_given = False

    def set_parameter(self, value):
        if value is None:
            self._parameter_given = False
            self._theta = None
        else: 
            self._parameter_given = True
            self._theta = value

            cos = np.cos(self._theta)
            sin = np.sin(self._theta)
            x, y, z = self._rotation_vector

            self._rotation_matrix = np.array([
                [
                    cos + x**2 * (1 - cos),
                    x * y * (1 - cos) - z * sin,
                    x * z * (1 - cos) + y * sin
                ],
                [
                    y * x * (1 - cos) + z * sin,
                    cos + y**2 * (1 - cos),
                    y * z * (1 - cos) - x * sin
                ],
                [
                    z * x * (1 - cos) - y * sin,
                    z * y * (1 - cos) + x * sin,
                    cos + z**2 * (1 - cos)
                ]
            ]).reshape((3,3))

            self._rotation_matrix_derivative = np.array([
                [
                    -sin + x**2 * sin,
                    x * y * sin - z * cos,
                    x * z * sin + y * cos
                ],
                [
                    y * x * sin + z * cos,
                    -sin + y**2 * sin,
                    y * z * sin - x * cos
                ],
                [
                    z * x * sin - y * cos,
                    z * y * sin + x * cos,
                    -sin + z**2 * sin
                ]
            ]).reshape((3,3))


    def function(self, vector: np.ndarray):
        if not self._parameter_given:
            raise ValueError("set_parameter has not been called")

        return self._rotation_matrix @ vector
    
    def derivative_WR_vector(self):
        if not self._parameter_given:
            raise ValueError("set_parameter has not been called")

        return self._rotation_matrix_derivative
    
    def derivative_WR_parameter(self):
        if not self._parameter_given:
            raise ValueError("set_parameter has not been called")

        return self._rotation_matrix