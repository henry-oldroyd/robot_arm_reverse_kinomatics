from transformation_compotision import Transformation_Composion
from differential_operations import Range_Limit_Vector
from math import pi, exp
import numpy as np

arm_initial_angles = (pi, pi/2, pi/2, pi/2)
arm_lengths = (1, 3, 2, 1)
arm_length_sums = tuple(sum(arm_lengths[:i+1]) for i in range(4))

class Robot_Arm:
    def __init__(self) -> None:
        self.angles = arm_initial_angles


        # I think the [:-1] may be an issue that needs debugging
        self.joint_positions = tuple(
            np.array([0, 0, l])
            for l in arm_length_sums[:-1]
        )

        self.grap_point_position = np.array([0, 0, arm_length_sums[-1]])
        self.base_position = np.array([0, 0, 0])

        # set up the grap point transformation
        self.grap_point_transformation_composition = Transformation_Composion()

        j_vector = np.array([0, 1, 0])
        k_vector = np.array([0, 0, 1])

        for joint_num in (3, 2, 1):
            self.grap_point_transformation_composition.add_rotation_about_vector(
                rotation_vector = j_vector,
                center_vector = self.joint_positions[joint_num-1],
                parameter_name = f"theta_j{joint_num}"
            )
        
        self.grap_point_transformation_composition.add_rotation_about_vector(
            rotation_vector = k_vector,
            center_vector = self.base_position,
            parameter_name = f"theta_j0"
        )


        

    def get_GP_position_at_angles(self, angles):
        parameters = {
            f"theta_j{j_num}": angle - initial_angle
            for j_num, angle, initial_angle in zip(range(4), angles, arm_initial_angles)
        }
        assert len(parameters) == 4, "The number of parameters should be the same as the number of joints"

        return self.grap_point_transformation_composition.apply_operations_to_vector(
            parameters = parameters,
            starting_vector = self.grap_point_position
        )
    



    def get_angles_for_GP_position(self, desired_GP_position):
        if len(desired_GP_position) != 3:
            raise ValueError("Desired displacement vector must have exactly 3 elements")
        if np.linalg.norm(desired_GP_position) > sum(arm_lengths):
            raise ValueError("The provided position is out of reach")
        

        angle_ranges = [2*pi, pi, pi, pi]
        angles_mapping = Range_Limit_Vector(
            ((0, maximum) for maximum in angle_ranges)
        ) 

        # # corresponds to each servo angle starging in middle of range
        # current_angles_precursors = np.zeros((4,))
        # current_angles = angles_mapping.map_vector_to_target_interval(current_angles_precursors)
        # assert all(val < 0.000_1 for val in (current_angles - self.angles))

        learning_rate = 0.1
        max_tries = 20
        max_updates = 500

        best_angles = None
        best_loss = 10**10



        for try_num in range(max_tries):
            if try_num == 0:
                # corresponds to each servo angle starging in middle of range
                current_angles_precursors = np.zeros((4,))
            else:
                current_angles_precursors = np.random.standard_normal((4,))
            
            current_angles = angles_mapping.map_vector_to_target_interval(current_angles_precursors)

            for update_index in range(max_updates):
                current_GP_position = self.get_GP_position_at_angles(current_angles)

                current_transformation_parameter_derivatives = self.grap_point_transformation_composition.get_parameter_derivatives() 
                current_angles_derivative = np.array([
                    current_transformation_parameter_derivatives[f"theta_j{j_num}"]
                    for j_num in range(4)
                ]).reshape((4, 3))

                angles_mapping_derivative = angles_mapping.derivative_of_mapping(current_angles_precursors)

                # current_angles_precursors += (1/2) * learning_rate * (
                #     angles_mapping_derivative
                #     @ current_angles_derivative
                #     @ (desired_GP_position - current_GP_position)
                # )

                current_angles_precursors += (-1/2) * learning_rate * (
                    angles_mapping_derivative
                    @ current_angles_derivative
                    @ (desired_GP_position - current_GP_position)
                )

                current_angles = angles_mapping.map_vector_to_target_interval(current_angles_precursors)


                loss = (desired_GP_position - current_GP_position).T @ (desired_GP_position - current_GP_position)
                
                # print(f"Iteration {update_index}:")
                # print(f"Current angles are:   {[round(value, 4) for value in current_angles]}")
                # print(f"Current position is:  {[round(value, 4) for value in current_GP_position]}")
                # print(f"Current loss is:  {loss}")      
            
            if try_num == 0 or loss < best_loss:
                best_loss = loss
                best_angles = current_angles


        return best_angles


