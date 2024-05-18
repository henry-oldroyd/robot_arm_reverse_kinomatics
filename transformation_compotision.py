import differential_operations as do
import numpy as np


class Transformation_Composion():
    def __init__(self) -> None:
        self._operations: list[do.Differential_Operation, str] = []
        self._required_parameter_names = set()

    def add_operation(self, operation: do.Differential_Operation, parameter_name: str = None):
        if parameter_name is not None:
            if parameter_name in self._required_parameter_names:
                raise ValueError("The parameter name is already taken")
            self._required_parameter_names.add(parameter_name)


        self._operations.append(
            [operation, parameter_name]
        )


    def add_rotation_about_vector(self, rotation_vector, center_vector, parameter_name):
        self.add_operation(do.Translate(-center_vector))
        self.add_operation(
            do.Rotation_About_Vector(rotation_vector),
            parameter_name = parameter_name,
        )
        self.add_operation(do.Translate(center_vector))

    def apply_operations_to_vector(self, parameters: dict, starting_vector) -> np.ndarray:
        msg = f"The provided parameter {list(parameters.keys())} don't match the parameters of the transformation {list(self._required_parameter_names)}"
        assert set(parameters.keys()) == set(self._required_parameter_names), msg
        assert len(self._operations) != 0, "No operations provided"

        image_vector = starting_vector
        for operation, parameter_name in self._operations:
            operation: do.Differential_Operation
            operation.set_parameter(
                value = parameters.get(parameter_name)
            )
            image_vector = operation.function(image_vector)

        return image_vector
    
    def clear_operations(self):
        self._operations.clear()
        self._image_vector = None

    def get_parameter_derivatives(self):
        parameter_derivatives = {}
        derivative_image_vector = np.ones((3, 1))
        for operation, parameter_name in reversed(self._operations):
            operation: do.Differential_Operation
            if parameter_name is not None:
                parameter_derivatives[parameter_name] = operation.derivative_WR_parameter() @ derivative_image_vector
            derivative_image_vector = operation.derivative_WR_vector() @ derivative_image_vector

        return parameter_derivatives
