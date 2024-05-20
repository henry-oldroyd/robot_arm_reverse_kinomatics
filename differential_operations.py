# import necessary finctions
import abc
import numpy as np
from math import exp
from functools import lru_cache


# this is an iterface for a differential transformation to be used in the tansformation composition class
# requires methods to compute vector image, derivatives and to set a traiable paramter
class Differentiable_Tranformation(abc.ABC):
    @abc.abstractmethod
    def function(self, X:np.ndarray):
        pass

    @abc.abstractmethod
    def derivative_WR_vector(self, X: np.ndarray=None):
        pass


# this object acts as a differential mapping function
# using sigmoid it maps any real value to a value in a specified interval
# this is used to limit the range of possible angles to within the capacities of the servo motors
class Range_Limiter_Scalar():
    # constructor takes in the minimum and maximum values of the interval
    def __init__(self, minimum, maximum):
        # validate that both arguments are ints of floats
        if not all(isinstance(x, (int, float)) for x in (minimum, maximum)):
            raise TypeError("Both minimum and maximum must be integers or floats")

        # cast to floats
        minimum, maximum = float(minimum), float(maximum)

        # validate that minimum is less than maximum
        if minimum >= maximum:
            raise ValueError("The minimum value must be less than the maximum value")

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
class Range_Limit_Vector(Differentiable_Tranformation):
    # constructor parameterised by each values corresponding interval
    def __init__(self, intervals):
        # validate that intervals is a list tuple or numpy array
        if not isinstance(intervals, (list, tuple, np.ndarray)):
            raise TypeError("Intervals must be a list, tuple or numpy array")
        # check that all emenent are lists tuples or numpy arrays
        if not all(isinstance(interval, (list, tuple, np.ndarray)) for interval in intervals):
            raise TypeError("Each element of intervals must be a list, tuple or numpy array")
        # check that all elements have length 2
        if not all(len(interval) == 2 for interval in intervals):
            raise ValueError("Each element of intervals must have length 2")

        # cast to tuple of tuples
        self.intervals = tuple(tuple(interval) for interval in intervals)
        self.vector_size = len(self.intervals)

        # create a list of Range_Limiter objects to map to coreesponding intervals
        self.mappings = tuple((Range_Limiter_Scalar(*interval) for interval in self.intervals))

    # this function maps the vector to one of controlled intervals
    def function(self, X):
        # element wise mapping of precursors to target intervals
        mapped_values = [
            mapping.map_scalar_to_target_interval(x)
            for x, mapping in zip(X, self.mappings)
        ]
        return np.array(mapped_values)
    
    # this is the matrix derivative of this operation
    def derivative_WR_vector(self, X):
        # produce a diagonal matrix of the derivatives of the mappings
        mapping_derivatives = [
            mapping.derivative_of_mapping(x)
            for x, mapping in zip(X, self.mappings)            
        ]
        return np.diag(mapping_derivatives)


# this is a subclass of Differentiable_Tranformation
# it includes a trainable parameter that can be set
# this affects the transformation to the image vector
# the derivative of the image vector with respect to the parameter can also be computed
class Parameterised_Differentiable_Tranformation(Differentiable_Tranformation):
    @abc.abstractmethod
    def set_parameter(self, parameter_value):
        pass
    
    @abc.abstractmethod
    def derivative_WR_parameter(self, X=None):
        pass


# this operation has no parameter
# it represents the translation of a vector
class Translate(Differentiable_Tranformation):
    # constructor takes in the translation vector
    def __init__(self, translation_vector: np.ndarray) -> None:
        # check that the translation vector is a numpy array
        if not isinstance(translation_vector, np.ndarray):
            raise TypeError("The translation vector must be a numpy array")

        self._translation_vector = translation_vector
        self._vector_length = len(translation_vector)

    def function(self, X: np.ndarray):
        return X + self._translation_vector
    
    # derivatives are identity matricies
    def derivative_WR_vector(self, X=None):
        # return identity matrix
        return np.eye(self._vector_length)


# this operation is a linear transformation in 3d (origin unaffected)
# it rotated points by theta about some vector (coudld be the unit vectors but doesn't have to be)
class Rotation_About_Vector_3D(Parameterised_Differentiable_Tranformation):
    # constructor takes in the vector you are rotating about
    def __init__(self, rotation_vector: np.ndarray) -> None:
        # check that the rotation vector is a numpy array
        if not isinstance(rotation_vector, np.ndarray):
            raise TypeError("The rotation vector must be a numpy array")
        # check that the rotation vector is of length 3
        if len(rotation_vector) != 3:
            raise ValueError("The rotation vector must have length 3")
        
        # check that the rotation vector has a non zero length
        rotation_vector_length = np.linalg.norm(rotation_vector)
    
        if rotation_vector_length == 0:
            raise ValueError("The rotation vector must have a non zero length")
        
        # normalise rotation vector to unit vector
        rotation_vector = (1/rotation_vector_length) * rotation_vector

        self._rotation_vector = rotation_vector
        self._parameter_given = False

    # the set parameter function is used to provide the value of theta. 
    # this value can be changed throughout the use of the object
    def set_parameter(self, parameter_value):
        # special case if parameter is None
        # resests object to not have a parameter
        if parameter_value is None:
            self._parameter_given = False
            self._theta = None
            return None

        # check that the parameter value is a float
        if not isinstance(parameter_value, float):
            raise TypeError("The parameter value must be a float (or none to remove parameter)")

        # when parameter given store it
        self._parameter_given = True
        self._theta = parameter_value

        # modify theta to be in the range of 0 to 2pi
        self._theta = self._theta % (2 * np.pi)

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
    def function(self, X: np.ndarray):
        # validate that X is a numpy array
        if not isinstance(X, np.ndarray):
            raise TypeError("The input vector must be a numpy array")
        # validate that X is of length 3
        if len(X) != 3:
            raise ValueError("The input vector must have length 3")

        if not self._parameter_given:
            raise ValueError("set_parameter has not been called")

        return self._rotation_matrix @ X
    
    # derivative of image vector with respect to input vector is just the matrix
    # must have parameter provided
    def derivative_WR_vector(self, X=None):
        if not self._parameter_given:
            raise ValueError("set_parameter has not been called")

        return self._rotation_matrix
    
    # derivative of the image vector with respect to the parameter is the other pecalculated matrix
    # must have parameter provided
    def derivative_WR_parameter(self, X=None):
        if not self._parameter_given:
            raise ValueError("set_parameter has not been called")
 
        return self._rotation_matrix_derivative
    

# composition of rotations and translations
class Centered_Rotaion_About_Vector_3D(Parameterised_Differentiable_Tranformation):
    
    # constructor takes in the rotation vector and center vector
    def __init__(self, rotation_vector: np.ndarray, center_vector: np.ndarray) -> None:
        # valiate both vectors by passing them to relevant objects with internal validation
        
        # translation operation to move center to origin for linear transformation and then back
        self._translation_center_to_origin = Translate(-center_vector)
        self._translation_origin_to_center = Translate(center_vector)

        # rotation operation to rotate about vector
        self._rotation = Rotation_About_Vector_3D(rotation_vector)

        self._parameter_given = False

    def set_parameter(self, value: float):
        # if value is None remove parameter
        if value is None:
            self._parameter_given = False
            self._rotation.set_parameter(None)
            return None
        
        # otherwise check its a float
        if not isinstance(value, float):
            raise TypeError("The parameter value must be a float (or none to remove parameter)")
        

        # set parameter for rotation operation
        self._rotation.set_parameter(value)
        self._parameter_given = True
    
    def function(self, X: np.ndarray):
        # apply the operations in sequence
        # these objects have internal validation for X
        X = self._translation_center_to_origin.function(X)
        X = self._rotation.function(X)
        X = self._translation_origin_to_center.function(X)
        return X
    
    def derivative_WR_vector(self, X=None):
        # don't need X
        # traslations will have identity derivative so this can be ignored to save computation (multiplicatictive identity)

        # return the rotation operation derivative
        return self._rotation.derivative_WR_vector(X)
    
    def derivative_WR_parameter(self, X=None):
        # don't need X
        # traslations will have identity derivative so this can be ignored to save computation (multiplicatictive identity)

        # return the rotation operation derivative
        return self._rotation.derivative_WR_parameter(X)