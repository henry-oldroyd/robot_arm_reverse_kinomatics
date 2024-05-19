import differential_operations as do
import numpy as np

# this class will represent a differential parameterised non linear transformation of a vector
# it will be composed of Differentiable_Tranformation and Parameterised_Differentiable_Tranformation
# it will be able to foreward propagate to take a vector and produce the image vector from this combined transformation
# this will require a dictionary of names parameters for each of the Parameterised_Differentiable_Tranformation
# it can also back propagate to find the pertial derivatives of the image vector with respect to each of the trainable paramters (used for SGD)
class Transformation_Composion():
    # constructor takes not arguments
    def __init__(self) -> None:
        self._operations: list[do.Differentiable_Tranformation] = []
        self._required_parameters: dict[int, str] = {}

    # this adds an operation to the operations array composing it with the others
    # if it is parameterised the name of the parameter is stored with the operation
    def add_operation(self, operation: do.Differentiable_Tranformation, parameter_name: str = None):
        # assert that a typed Differentiable_Tranformation is given
        if not isinstance(operation, do.Differentiable_Tranformation):
            raise TypeError("The operation parameter must inherit form the Differentiable_Tranformation interface")


        # handle checks if its paramertised
        if isinstance(operation, do.Parameterised_Differentiable_Tranformation):
            # ensure a parameter name is provided
            if parameter_name is None:
                raise ValueError("The Parameterised_Differentiable_Tranformation must have an acompanying parameter name")
            # its a string
            if not isinstance(parameter_name, str):
                raise TypeError("Parameter name must be of type string")
            # and that it hasn't already been taken by another operation
            if parameter_name in self._required_parameters.values(): 
                raise ValueError("The parameter name is already taken, it must be unique")
            
            # now assume name is valid and add it to the record

        self._operations.append(operation)

        # add it as a pair with key equal to the index of the operation
        if isinstance(operation, do.Parameterised_Differentiable_Tranformation):
            self._required_parameters[len(self._operations)-1] = parameter_name


    # this function will compute and add multiple transformations to allow for rotation about a vector with a center not at the origin
    def add_rotation_about_vector(self, rotation_vector, center_vector, parameter_name):
        # translate center_vector to orign
        self.add_operation(
            do.Translate(-center_vector)
        )
        # complete parameterised rotation about vector centered at orignin (linear transformation)
        self.add_operation(
            do.Rotation_About_Vector(rotation_vector),
            parameter_name,
        )
        # translate origin back to center vector 
        self.add_operation(
            do.Translate(center_vector)
        )


    # this method acts as the foreward propagation
    # it can compute the image of a starting vecotor given required parameters
    def apply_operations_to_vector(self, provided_parameters: dict, starting_vector) -> np.ndarray:

        # check that the provided parameters match the required parameters
        names_of_provided_parameters = set(provided_parameters.keys()) 
        names_of_required_parameters = set(self._required_parameters.values())
        if names_of_provided_parameters != names_of_required_parameters:
            raise ValueError(
                f"The provided parameter {sorted(names_of_provided_parameters)} don't match the parameters of the transformation {sorted(names_of_required_parameters)}"
            )

        # apply the operations iteratively to the starting vector
        image_vector = starting_vector

        # iterate through the operations
        for operation_i, operation in enumerate(self._operations):
            # check if the operation is parameterised
            # if so get the parameter form provided parameters and set it
            if isinstance(operation, do.Parameterised_Differentiable_Tranformation):
                parameter_name = self._required_parameters[operation_i]

                operation.set_parameter(
                    value = provided_parameters[parameter_name]
                )
            
            # apply the operation to the image vector
            image_vector = operation.function(image_vector)

        return image_vector
    
    # clear the operations array
    def clear_operations(self):
        self._operations.clear()
        self._image_vector = None

    # this method will return the partial derivatives of the image vector with respect to each of the named parameters of this composite transformation
    def get_parameter_derivatives(self):
        # construct a dictionary to store the derivatives
        parameter_derivatives = {}
        # starting derivative is multiplicative identiy vector of ones
        derivative_intermediate_image_vector = np.ones((3, 1))
        
        # iterator for operations and indexes in reverse
        reverse_operations_iterator = reversed(list(enumerate(self._operations)))
        # loop through the operations in reverse
        for operation_i, operation in reverse_operations_iterator:
            # if this operation is parameterised get derivative with respect to the parameter
            if isinstance(operation, do.Parameterised_Differentiable_Tranformation):
                parameter_name = self._required_parameters[operation_i]
                # take product of derivative of this operations image with respect to parameter and derivative of final image with respect to this operations image
                parameter_derivatives[parameter_name] = operation.derivative_WR_parameter() @ derivative_intermediate_image_vector
            
            # update intermediate image vector by taking product of derivative of this operation's image with respect to input and final imgae with respect to this operation's image
            derivative_intermediate_image_vector = operation.derivative_WR_vector() @ derivative_intermediate_image_vector

        # return the dictionary of parameter derivatives
        return parameter_derivatives
