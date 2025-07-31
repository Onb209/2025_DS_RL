import numpy as np
import gymnasium as gym
import os

# The observation space is a `Box(-Inf, Inf, (17,), float64)` where the elements are as follows:
# | Num | Observation                                        | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
# | --- | -------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
# | 0   | x-coordinate of the torso                          | -Inf | Inf | rootz                            | slide | position (m)             |
# | 1   | z-coordinate of the torso (height of Walker2d)     | -Inf | Inf | rootz                            | slide | position (m)             |
# | 2   | angle of the torso                                 | -Inf | Inf | rooty                            | hinge | angle (rad)              |
# | 3   | angle of the thigh joint                           | -Inf | Inf | thigh_joint                      | hinge | angle (rad)              |
# | 4   | angle of the leg joint                             | -Inf | Inf | leg_joint                        | hinge | angle (rad)              |
# | 5   | angle of the foot joint                            | -Inf | Inf | foot_joint                       | hinge | angle (rad)              |
# | 6   | angle of the left thigh joint                      | -Inf | Inf | thigh_left_joint                 | hinge | angle (rad)              |
# | 7   | angle of the left leg joint                        | -Inf | Inf | leg_left_joint                   | hinge | angle (rad)              |
# | 8   | angle of the left foot joint                       | -Inf | Inf | foot_left_joint                  | hinge | angle (rad)              |
# | 9   | velocity of the x-coordinate of the torso          | -Inf | Inf | rootx                            | slide | velocity (m/s)           |
# | 10  | velocity of the z-coordinate (height) of the torso | -Inf | Inf | rootz                            | slide | velocity (m/s)           |
# | 11  | angular velocity of the angle of the torso         | -Inf | Inf | rooty                            | hinge | angular velocity (rad/s) |
# | 12  | angular velocity of the thigh hinge                | -Inf | Inf | thigh_joint                      | hinge | angular velocity (rad/s) |
# | 13  | angular velocity of the leg hinge                  | -Inf | Inf | leg_joint                        | hinge | angular velocity (rad/s) |
# | 14  | angular velocity of the foot hinge                 | -Inf | Inf | foot_joint                       | hinge | angular velocity (rad/s) |
# | 15  | angular velocity of the thigh hinge                | -Inf | Inf | thigh_left_joint                 | hinge | angular velocity (rad/s) |
# | 16  | angular velocity of the leg hinge                  | -Inf | Inf | leg_left_joint                   | hinge | angular velocity (rad/s) |
# | 17  | angular velocity of the foot hinge                 | -Inf | Inf | foot_left_joint                  | hinge | angular velocity (rad/s) |

# The action space is a `Box(-1, 1, (6,), float32)`. An action represents the torques applied at the hinge joints.
# | Num | Action                                 | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
# |-----|----------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
# | 0   | Torque applied on the thigh rotor      | -1          | 1           | thigh_joint                      | hinge | torque (N m) |
# | 1   | Torque applied on the leg rotor        | -1          | 1           | leg_joint                        | hinge | torque (N m) |
# | 2   | Torque applied on the foot rotor       | -1          | 1           | foot_joint                       | hinge | torque (N m) |
# | 3   | Torque applied on the left thigh rotor | -1          | 1           | thigh_left_joint                 | hinge | torque (N m) |
# | 4   | Torque applied on the left leg rotor   | -1          | 1           | leg_left_joint                   | hinge | torque (N m) |
# | 5   | Torque applied on the left foot rotor  | -1          | 1           | foot_left_joint                  | hinge | torque (N m) |

class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, render_mode="human", bump_practice=False, bump_challenge=False):
        if bump_challenge:
            env = gym.make(
                "Walker2d-v5",
                xml_file=os.getcwd() + "/asset/custom_walker2d_bumps.xml",
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
                frame_skip = 10,
                healthy_z_range=(0.5, 10.0))
        elif bump_practice:
            env = gym.make(
                "Walker2d-v5",
                xml_file=os.getcwd() + "/asset/custom_walker2d_bumps_practice.xml",
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
                frame_skip = 10,
                healthy_z_range=(0.5, 10.0))
        else:
            env = gym.make(
                "Walker2d-v5",
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
                frame_skip = 10)
        
        super().__init__(env)
        
        ## change observation space according to the new observation
        obs, _ = self.reset()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(obs),), dtype=np.float64)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        custom_obs = self.custom_observation(obs)
        return custom_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        custom_obs = self.custom_observation(obs)
        custom_reward = self.custom_reward(obs, reward)
        custom_terminated = self.custom_terminated(terminated, obs)
        custom_truncated = self.custom_truncated(truncated)
        return custom_obs, custom_reward, custom_terminated, custom_truncated, info

    def custom_terminated(self, terminated, obs):
        # TODO: Implement your own termination condition
        return terminated
    
    def custom_truncated(self, truncated):
        # TODO: Implement your own truncation condition
        return truncated

    def custom_observation(self, obs):
        # TODO: Implement your own observation
        return obs

    def custom_reward(self, obs, original_reward):
        # TODO: Implement your own reward
        return original_reward

## Test Rendering
if __name__ == "__main__":
    env = CustomEnvWrapper()
    obs = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated:
            obs = env.reset()
