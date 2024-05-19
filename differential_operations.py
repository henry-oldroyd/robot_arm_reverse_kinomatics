# import necessary finctions
import abc
import numpy as np
from math import exp
from functools import lru_cache


# this object acts as a differential mapping function
# using sigmoid it maps any real value to a value in a specified interval
# this is used to limit the range of possible angles to within the capacities of the servo motors
class Range_Limiter():
    # constructor takes in the minimum and maximum values of the interval
    def __init__(self, minimum, maximum):
        self.min, self.max = minimum, maximum

    # use small cache as values are reused for the derivative
    @lru_cache(10)
    def sigmoid(self, x): 
        return 1/(1+exp(-x))
    
    # derivative of the sigmoid function
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    # simple scaling to map sigmoid range of (0,1) to (min, max)
    def map_scalar_to_target_interval(self, x):
        return (
            (self.max - self.min) 
            * self.sigmoid(x)
        ) + self.min

    # derivate of scaled value with respect to input value
    # differentiated with the chain rule
    def derivative_of_mapping(self, x):
        return (
           (self.max - self.min) 
            * self.sigmoid(x)
            * self.sigmoid_derivative(x) 
        )


# this is the vectorisation of the previous class
# it takes in a vector of any real values
# it maps each one to a different interval prodivided to the construtor
# it provided the matrix derivative of the output vecotor with respect to the input vector
class Range_Limit_Vector():
    # constructor parameterised by each values corresponding interval
    def __init__(self, intervals):
        self.intervals = tuple(intervals)
        self.vector_size = len(self.intervals)

        # create a list of Range_Limiter objects to map to coreesponding intervals
        self.mappings = tuple((Range_Limiter(*interval) for interval in self.intervals))

    # this function maps the vector to one of controlled intervals
    def map_vector_to_target_interval(self, X):
        # element wise mapping of precursors to target intervals
        mapped_values = [
            mapping.map_scalar_to_target_interval(x)
            for x, mapping in zip(X, self.mappings)
        ]
        return np.array(mapped_values)
    
    # this is the matrix derivative of this operation
    def derivative_of_mapping(self, X):
        # produce a diagonal matrix of the derivatives of the mappings
        mapping_derivatives = [
            mapping.derivative_of_mapping(x)
            for x, mapping in zip(X, self.mappings)            
        ]
        return np.diag(mapping_derivatives)


# this is an iterface for a differential transformation to be used in the tansformation composition class
# requires methods to compute vector image, derivatives and to set a traiable paramter
class Differentiable_Tranformation(abc.ABC):
    @abc.abstractmethod
    def function(vector: np.ndarray):
        pass

    @abc.abstractmethod
    def derivative_WR_vector():
        pass


# this is a subclass of Differentiable_Tranformation
# it includes a trainable parameter that can be set
# this affects the transformation to the image vector
# the derivative of the image vector with respect to the parameter can also be computed
class Parameterised_Differentiable_Tranformation(Differentiable_Tranformation):
    @abc.abstractmethod
    def derivative_WR_parameter():
        pass
    
    @abc.abstractmethod
    def derivative_WR_parameter():
        pass


# this operation has no parameter
# it represents the translation of a vector
class Translate(Differentiable_Tranformation):
    # constructor takes in the translation vector
    def __init__(self, translation_vector) -> None:
        self._translation_vector = translation_vector
        self._vector_length = len(translation_vector)

    def function(self, vector: np.ndarray):
        return vector + self._translation_vector
    
    # derivatives are identity matricies
    def derivative_WR_vector(self):
        # return identity matrix
        return np.eye(self._vector_length)
    

# this operation is a linear transformation in 3d (origin unaffected)
# it rotated points by theta about some vector (coudld be the unit vectors but doesn't have to be)
class Rotation_About_Vector(Parameterised_Differentiable_Tranformation):
    # constructor takes in the vector you are rotating about
    def __init__(self, rotation_vector) -> None:
        self._rotation_vector = rotation_vector
        self._parameter_given = False

    # the set parameter function is used to provide the value of theta. 
    # this value can be changed throughout the use of the object
    def set_parameter(self, value):
        # special case if parameter is None
        # resests object to not have a parameter
        if value is None:
            self._parameter_given = False
            self._theta = None
            return None

        # when parameter given store it
        self._parameter_given = True
        self._theta = value

        # calculate the rotation matrix and its derivative (from website)
        cos = np.cos(self._theta)
        sin = np.sin(self._theta)
        x, y, z = self._rotation_vector

        # this is the rotation matrix for 3d about a vector
        # https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
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

        # this is the partial derivative of the roation matrix with respect to theta evaluated at theta
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


    # apply linear map of rotation matrix to vector
    # must have parameter provided
    def function(self, vector: np.ndarray):
        if not self._parameter_given:
            raise ValueError("set_parameter has not been called")

        return self._rotation_matrix @ vector
    
    # derivative of image vector with respect to input vector is just the matrix
    # must have parameter provided
    def derivative_WR_vector(self):
        if not self._parameter_given:
            raise ValueError("set_parameter has not been called")


        # was set to return derivative matrix and appreared to work?
        # return self._rotation_matrix_derivative


        return self._rotation_matrix
    
    # derivative of the image vector with respect to the parameter is the other pecalculated matrix
    # must have parameter provided
    def derivative_WR_parameter(self):
        if not self._parameter_given:
            raise ValueError("set_parameter has not been called")

        # was set to return rotation matrix and appreared to work?
        # return self._rotation_matrix

        
        return self._rotation_matrix_derivative