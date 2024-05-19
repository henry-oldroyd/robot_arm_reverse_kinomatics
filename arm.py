# maths to check

# is angle to angle change accounted for in derivative calculation, propably identity so doens't matter?
# why does the formula need adjusting by -1 multiplier


from transformation_compotision import Transformation_Composion
from differential_operations import Range_Limit_Vector
from math import pi
import numpy as np

class Robot_Arm:
    # parameters are known vectors of joints at a certain combination of angles
    def __init__(self,
        starting_angles: list[float], 
        angle_intervals: list[tuple[float, float]], 
        starting_joint_position_vectors: list[np.ndarray], 
        starting_base_position_vector: np.ndarray, 
        starting_grab_point_position_vector: np.ndarray,
        total_arm_reach: float,
    ) -> None:
        # check that the total arm reach is a positive float
        if not isinstance(total_arm_reach, float):
            raise TypeError("The total arm reach must be of type float")
        if total_arm_reach <= 0:
            raise ValueError("The total arm reach must be a positive number")
        
        # assign to property
        self._total_arm_reach = total_arm_reach
        # check that angles, angle intervals and joint positions are all lists, tuples or numpy arrays
        if not all(isinstance(val, (list, tuple, np.ndarray)) for val in (starting_angles, angle_intervals, starting_joint_position_vectors)):
            raise TypeError("Angles, angle intervals and joint positions must be lists, tuples or numpy arrays")

        # check that the length of angles and angle intervals match the length of joints + 1 for the base
        if len(starting_angles) != len(angle_intervals) != len(starting_joint_position_vectors) + 1:
            raise ValueError("The length of angles and angle intervals should be equal to the length of joint positions + 1 for the base")

        # create variable to count number of joints
        self._num_joints = len(starting_joint_position_vectors)

        # check arm angles is a list of floats
        if not all(isinstance(angle, float) for angle in starting_angles):
            raise TypeError("All angles must be of type float")
        
    
        # check that the number of angles matches the number of joints and base servo
        if len(starting_angles) != len(starting_joint_position_vectors) + 1:
            raise ValueError("The number of angles should be equal to the number of joint positions + 1")

        
        # check that all joint vectors, base position, and grab point position are 3D coordinates
        coordinates_to_check = {
            f'joint_{i+1}': vector
            for i, vector in enumerate(starting_joint_position_vectors)
        }
        coordinates_to_check["base"] = starting_base_position_vector
        coordinates_to_check["grab_point"] = starting_grab_point_position_vector

        # check that all vectors are numpy arrays of shape (3,)
        for key, vector in coordinates_to_check.items():
            if not isinstance(vector, np.ndarray):
                raise TypeError(f"{key} position vector must be a numpy array")
            if vector.shape != (3,):
                raise ValueError(f"{key} position vector must have shape (3,)")

        # assign position vectors parameters to properties
        self._starting_angles = starting_angles
        self._starting_joint_position_vectors = starting_joint_position_vectors
        self._starting_base_position_vector = starting_base_position_vector
        self._grap_point_position = starting_grab_point_position_vector


        # validate the angle intervals
        for interval in angle_intervals:
            if not isinstance(interval, (list, tuple)):
                raise TypeError("Each angle interval must be a list or tuple")
            if len(interval) != 2:
                raise ValueError("Each angle interval must contain exactly 2 elements")
            if not all(isinstance(angle, float) for angle in interval):
                raise TypeError("Both elements in each angle interval must be of type float")
            if not interval[0] < interval[1]:
                raise ValueError("The first angle in each interval must be strictly less than the second")

        # assign the angle intervals to a property
        self._angle_intervals = angle_intervals

        # create a Range_Limit_Vector object to map the angles to the intervals
        self._angle_interval_mapping = Range_Limit_Vector(angle_intervals)

        # set up the grap point transformation as a blank composition
        self._grap_point_transformation_composition = Transformation_Composion()


        # build up the transformation composition by adding joints in reverse order then base 
        # move base last so joints can rotate about 

        # use unit vectors for the rotation axes
        j_vector = np.array([0, 1, 0])
        k_vector = np.array([0, 0, 1])

        # iterate through the joint indexes in reverse order 
        joint_indexes_reversed = reversed(range(len(starting_joint_position_vectors)))
        for joint_i in joint_indexes_reversed:
            # add a rotation about the j axis for each joint to the transformation composition
            self._grap_point_transformation_composition.add_rotation_about_vector(
                rotation_vector=j_vector,
                center_vector=starting_joint_position_vectors[joint_i],
                parameter_name=f"theta_j{joint_i+1}"
            )

        # add a rotation about the k axis for the base servo to the transformation composition
        self._grap_point_transformation_composition.add_rotation_about_vector(
            rotation_vector=k_vector,
            center_vector=starting_base_position_vector,
            parameter_name=f"theta_j0"
        )

        
    # this method will return the position of the grap point given the angles of the arm
    def get_GP_position_at_angles(self, angles: np.ndarray) -> np.ndarray:
        # check type of angles
        if not isinstance(angles, np.ndarray):
            raise TypeError("Angles must be a numpy array")
        # check number of angles is base + num joints
        if len(angles) != self._num_joints + 1:
            raise ValueError("The number of angles should be equal to the number of joint positions + 1")
        
        # for each angle and angle interval check that the angle is within the interval
        for angle, interval in zip(angles, self._angle_intervals):
            if not interval[0] <= angle <= interval[1]:
                raise ValueError(f"Angle {angle} is not within the interval {interval}")

        # construct the parameters dictionary for the transformation composition
        # the angle change is angle - initial_angle
        parameters = {
            f"theta_j{j_num}": angle - initial_angle
            for j_num, angle, initial_angle 
            in zip(range(self._num_joints + 1), angles, self._starting_angles)
        } 

        # apply the transformation composition to the starting grap point position to get image
        # return new image
        return self._grap_point_transformation_composition.apply_operations_to_vector(
            provided_parameters = parameters,
            starting_vector = self._grap_point_position
        )
    


    # this method will return the angles of the arm given the position of the grap point
    # this is an optimisation problem so we will use gradient descent
    def get_angles_for_GP_position(self, desired_GP_position: np.ndarray):
        # check that the desired position is a numpy array of shape (3,)
        if not isinstance(desired_GP_position, np.ndarray):
            raise TypeError("The desired position must be a numpy array")
        if desired_GP_position.shape != (3,):
            raise ValueError("The desired position must have shape (3,)")

        if np.linalg.norm(desired_GP_position) > self._total_arm_reach:
            raise ValueError("The provided position is out of reach")


        # hyperparameters for the gradient descent search for angles
        # high learning rate numerically stable for this problem
        learning_rate = 0.5
        # number of random starting angles improved with SGD before best taken
        max_tries = 1
        # the number of updates with SGD for each set of starting angles
        max_updates = 200

        # record the best angles and loss so far in tries
        best_angles = None
        best_loss = 10**10



        # iterate through the number of tries
        for try_num in range(max_tries):
            # if first try use half way angles, else use random normal
            if try_num == 0:
                # corresponds to each servo angle starging in middle of range
                current_angles_precursors = np.zeros((self._num_joints + 1,))
            else:
                current_angles_precursors = np.random.standard_normal((self._num_joints + 1,))
           
            # map the angles to the target interval
            current_angles = self._angle_interval_mapping.map_vector_to_target_interval(current_angles_precursors)

            # for each SGD update
            for update_index in range(max_updates):
                # get the current position of the grap point at current angles
                current_GP_position = self.get_GP_position_at_angles(current_angles)

                # get the derivative of the current position with respect to the angles
                current_transformation_parameter_derivatives = self._grap_point_transformation_composition.get_parameter_derivatives() 
                # convert paramerter derivatives dict to numpy array

                current_angles_derivative = np.array([
                    current_transformation_parameter_derivatives[f"theta_j{j_num}"]
                    for j_num in range(self._num_joints + 1)
                ]).reshape((self._num_joints + 1, 3))

                # get the derivative of the mapping from precursors to angles
                angles_mapping_derivative = self._angle_interval_mapping.derivative_of_mapping(current_angles_precursors)

                # apply chain rule to get dl/dprecursors
                derivative_of_loss_WR_precursors = (-1/2) * (
                    angles_mapping_derivative
                    @ current_angles_derivative
                    @ (desired_GP_position - current_GP_position)
                )

                # update the precursors with the gradient descent step
                current_angles_precursors -= learning_rate * derivative_of_loss_WR_precursors

                # get current angles from new precursors
                current_angles = self._angle_interval_mapping.map_vector_to_target_interval(current_angles_precursors)

                # get current loss from new precursors
                loss = (desired_GP_position - current_GP_position).T @ (desired_GP_position - current_GP_position)

                # # print out the current state of the optimisation algorithm             
                # print(f"Iteration {update_index}:")
                # print(f"Current angles are:   {[round(value, 4) for value in current_angles]}")
                # print(f"Current position is:  {[round(value, 4) for value in current_GP_position]}")
                # print(f"Current loss is:  {loss}")      
            
            # if the loss is the best so far, update the best loss and angles
            if try_num == 0 or loss < best_loss:
                best_loss = loss
                best_angles = current_angles

        # return the best angles found
        return best_angles


# this represents the leipzig robot arm specifically
class Leipzig_Robot_Arm(Robot_Arm):
    def __init__(self) -> None:
        # this is an arbitrary set of initial angles and arm lengths
        # these should be replaces with known infomation about the arms position at specific angles
        arm_initial_angles = (pi, pi/2, pi/2, pi/2)

        arm_max_angles = (2*pi, pi, pi, pi)
        arm_angle_intervals = tuple((0.0, max_angle) for max_angle in arm_max_angles)

        # compute position vectors of all joints and the grap point
        arm_lengths = (1, 3, 2, 1)
        arm_length_sums = tuple(sum(arm_lengths[:i+1]) for i in range(4))

        joint_positions = tuple(
            np.array([0, 0, l])
            for l in arm_length_sums[:-1]
        )
        grap_point_position = np.array([0, 0, arm_length_sums[-1]])
        base_position = np.array([0, 0, 0])

        # compute the total arm reach
        total_arm_reach = float(arm_length_sums[-1])



        super().__init__(
            starting_angles = arm_initial_angles,
            angle_intervals = arm_angle_intervals,
            starting_joint_position_vectors = joint_positions,
            starting_base_position_vector = base_position,
            starting_grab_point_position_vector = grap_point_position,
            total_arm_reach = total_arm_reach
        )