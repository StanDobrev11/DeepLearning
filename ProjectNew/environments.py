from typing import Tuple, Optional, Any, Dict, List

import gymnasium as gym
import pygame
from gymnasium import spaces
from gymnasium.core import ObsType
from gymnasium.spaces.utils import flatten, flatten_space

import numpy as np

from vessels import OwnShip, Target, StaticObject


class MarineEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    OWN_SHIP_PARAMS: List[str] = []
    WP_PARAMS: List[str] = []
    TARGET_PARAMS: List[str] = []

    def __init__(
            self,
            initial_lat: float = 0.0,
            initial_lon: float = 0.0,
            env_range: int = 30,
            timescale: int = 1,
            continuous: bool = False,
            max_turn_angle: int = 10,
            max_speed_change: float = 0.1,
            waypoint_reach_threshold: float = 0.2,
            render_mode=None,
            training_stage: int = 1,  # defines different training stages, 0 for no training
            total_targets: Optional[int] = None
    ):
        super(MarineEnv, self).__init__()
        self.env_range = env_range
        self.step_counter = 0
        self.training_stage = training_stage
        self.total_targets = total_targets

        # initialize the environment bounds
        self.lat_bounds: Tuple[float, float] = (initial_lat, initial_lat + env_range / 60)
        self.lon_bounds: Tuple[float, float] = (initial_lon, initial_lon + env_range / 60)

        # define the timescale and scaling of real word time to simulation time
        self.timescale = timescale

        # initialize action space
        if continuous:
            # the agent can continuously adjust course and speed
            self.max_turn_angle = max_turn_angle  # scaling the turn
            self.max_speed_change = max_speed_change  # scaling the speed change
            self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(5)

        # define the observation space
        self.observation_space = flatten_space(self._define_observation_space())

        # initialize the state
        self.observation = None

        # initialize own ship
        self.own_ship = OwnShip()

        # initialize the wp and wp reach threshold
        self.waypoint = StaticObject()  # 2 elements, latitude and longitude
        self.waypoint_reach_threshold = waypoint_reach_threshold  # in nautical miles, terminates an episode
        self.waypoints = []

        # extract params
        self.OWN_SHIP_PARAMS = [key for key in self._define_observation_space()['own_ship'].spaces.keys()][:2]
        self.WP_PARAMS = [key for key in self._define_observation_space()['own_ship'].spaces.keys()][2:]
        self.TARGET_PARAMS = [key for key in self._define_observation_space()['targets'][0].spaces.keys()]

        # pygame setup
        assert render_mode is None or render_mode in self.metadata["render_modes"], \
            f"Invalid render_mode: {render_mode}. Available modes: {self.metadata['render_modes']}"
        self.render_mode = render_mode

        self.window_size = 600  # Pixels for visualization
        self.scale = self.window_size / ((self.lat_bounds[1] - self.lat_bounds[0]) * 60)  # Pixels per NM
        self.window = None
        self.clock = None
        self.vessel_size = 5  # Vessel radius in pixels

    def _define_observation_space(self):
        return spaces.Dict({
            'own_ship': spaces.Dict({
                'course': spaces.Box(low=0, high=360, shape=(), dtype=np.float32),
                'speed': spaces.Box(low=-10, high=20, shape=(), dtype=np.float32),
                'wp_distance': spaces.Box(low=0, high=self.env_range, shape=(), dtype=np.float32),
                'wp_eta': spaces.Box(low=0, high=300, shape=(), dtype=np.float32),
                'wp_relative_bearing': spaces.Box(low=-180, high=180, shape=(), dtype=np.float32),
                'wp_target_eta': spaces.Box(low=0, high=300, shape=(), dtype=np.float32),
            }),
            'targets': spaces.Tuple([
                spaces.Dict({
                    'bcr': spaces.Box(low=-50, high=50, shape=(), dtype=np.float32),
                    'cpa': spaces.Box(low=0, high=50, shape=(), dtype=np.float32),
                    'target_course': spaces.Box(low=0, high=360, shape=(), dtype=np.float32),
                    'target_distance': spaces.Box(low=0, high=self.env_range, shape=(), dtype=np.float32),
                    'target_relative_bearing': spaces.Box(low=-180, high=180, shape=(), dtype=np.float32),
                    'target_relative_course': spaces.Box(low=0, high=360, shape=(), dtype=np.float32),
                    'target_relative_speed': spaces.Box(low=0, high=50, shape=(), dtype=np.float32),
                    'target_speed': spaces.Box(low=-10, high=20, shape=(), dtype=np.float32),
                    'tbc': spaces.Box(low=-10, high=100, shape=(), dtype=np.float32),
                    'tcpa': spaces.Box(low=-60, high=100, shape=(), dtype=np.float32),
                }) for _ in range(3)  # assuming 3 targets are considered dangerous
            ])
        })

    @property
    def wp_distance(self) -> float:
        return self.own_ship.calculate_distance(self.waypoint)

    @property
    def wp_eta(self) -> float:
        # will be recalculated when calling reset()
        return 60 * self.wp_distance / self.own_ship.speed

    @property
    def wp_relative_bearing(self) -> float:
        return self.own_ship.calculate_relative_bearing(self.waypoint)

    @property
    def wp_target_eta(self) -> float:
        # will ALWAYS be calculated when handling the state
        return 0.0

    def step(self, action) -> tuple[ObsType, float, bool, bool, dict[str, int]]:

        # increment the step counter
        self.step_counter += 1

        # extract own ship params from the current state
        course, speed, last_wp_distance, last_wp_eta, last_wp_relative_bearing, last_tgt_eta = self.observation[:6]

        # update the params based on action
        if isinstance(self.action_space, spaces.Discrete):
            course_change = 0
            speed_change = 0
            if action == 0:  # do nothing, keep course and speed
                pass
            elif action == 1:  # turn port (left) by 1 degree
                course_change = - 1
            elif action == 2:  # turn starboard (right) by 1 degree
                course_change = 1
            elif action == 3:  # reduce speed
                speed_change = -0.1
            elif action == 4:  # increase speed
                speed_change = 0.1
        else:
            course_change = np.clip(action[0] * self.max_turn_angle, -self.max_turn_angle, self.max_turn_angle)
            speed_change = action[1] * self.max_speed_change

        # update and apply own ship parameters
        self.own_ship.course = (course + course_change) % 360
        self.own_ship.speed += speed_change

        # update next position
        self.own_ship.update_position(time_interval=self.timescale, clip_lat=self.lat_bounds, clip_lon=self.lon_bounds)

        # generate own state data
        own_ship_data = self._generate_own_ship_data()
        own_ship_data['wp_target_eta'] = last_tgt_eta - self.timescale

        if self.own_ship.detected_targets:
            # Move/update all detected targets
            for target in self.own_ship.detected_targets:
                target.update_position(time_interval=self.timescale, clip_lat=self.lat_bounds, clip_lon=self.lon_bounds)
                self.own_ship.update_target(target)

            # Sort detected targets by dangerous coefficient (CPA ** 2 * TCPA) and take the top 3
            self.own_ship.dangerous_targets = sorted(
                self.own_ship.detected_targets, key=lambda x: x.cpa ** 2 * x.tcpa
            )[:3]

            # initialize empty targets list with zero-filled entries
            targets_data = self._generate_zero_target_data(3)

            # fill in detected targets (up to 3)
            for i, target in enumerate(self.own_ship.dangerous_targets):
                target.is_dangerous = True
                targets_data[i] = self._generate_actual_target_data(target)

        else:
            # no detected targets, return all-zero target data
            targets_data = self._generate_zero_target_data(3)

        # construct the final observation
        raw_observation = {
            "own_ship": own_ship_data,
            "targets": targets_data
        }

        previous_obs = self.observation
        current_obs = self._flatten_observation(raw_observation)

        reward, terminated, truncated, info = self.calculate_reward(previous_obs, current_obs)

        # assign the current observation
        self.observation = current_obs

        return self.observation, reward, terminated, truncated, info

    def calculate_reward(self, previous_obs: ObsType, current_obs: ObsType) -> tuple[float, bool, bool, dict[str, int]]:
        """ method to calculate the reward """

        previous_data = self._generate_observation_dict(previous_obs)
        current_data = self._generate_observation_dict(current_obs)

        # terminated - the episode ends because the agent has reached a goal or violated an environment rule
        # (e.g., collision, reaching the waypoint, ect).
        # truncated - the episode ends due to external constraints like a maximum time step limit,
        # regardless of whether the agent reached a goal or not.
        terminated, truncated = False, False
        info = {'total_steps': self.step_counter}
        reward = 0.0

        # reward for wp tracking, training stage 1
        if self.training_stage == 1:
            # reaching the wp -> large reward and episode termination
            if self.wp_distance < self.waypoint_reach_threshold:
                return 100.0, True, False, info

            # Distance-Based Reward or Penalty
            previous_wp_distance = previous_data['own_ship']['wp_distance']
            current_wp_distance = current_data['own_ship']['wp_distance']

            distance_change = previous_wp_distance - current_wp_distance
            # reward += max(-1.0, distance_change * 10)  # Reward proportional to distance improvement
            reward += distance_change * 10 if distance_change > 0 else -1

            # bearing alignment reward
            current_wp_relative_bearing = current_data['own_ship']['wp_relative_bearing']
            if abs(current_wp_relative_bearing) <= 1:
                reward += 5.0
            elif abs(current_wp_relative_bearing) <= 5:
                reward += 2.0  # Strong reward for near-perfect alignment
            elif abs(current_wp_relative_bearing) <= 15:
                reward += 1.0  # Moderate reward for good alignment
            elif abs(current_wp_relative_bearing) <= 30:
                reward += 0.5  # Small reward for decent alignment
            else:
                reward -= 0.5  # Penalty for poor alignment


            # course change reward
            previous_relative_bearing = previous_data['own_ship']['wp_relative_bearing']
            current_relative_bearing = current_data['own_ship']['wp_relative_bearing']

            previous_course = previous_data['own_ship']['course']
            current_course = current_data['own_ship']['course']

            # Compute previous and current alignment error (how well the course aligns with the target)
            previous_alignment_error = abs(previous_relative_bearing - previous_course)
            current_alignment_error = abs(current_relative_bearing - current_course)

            # Check if alignment error increased (ship turned away)
            if current_alignment_error > previous_alignment_error:
                reward -= (current_alignment_error - previous_alignment_error) * 0.1  # Penalty for moving away
            else:
                reward += (previous_alignment_error - current_alignment_error) * 0.1  # Reward for improving alignment


        # TODO reward for training stage 2, one target
        # TODO reward for training stage 3, two targets
        # TODO reward for training stage 4, three targets

        # reward for keeping ETA steady
        current_eta = current_data['own_ship']['wp_eta']
        target_eta = current_data['own_ship']['wp_target_eta']

        if -6 <= current_eta - target_eta <= 6:
            reward += 0.1
        else:
            reward -= 0.5

        # if target_eta < -6:
        #     truncated = True
        #     return reward, terminated, truncated, info

        # Penalty for Going Out of Bounds
        out_of_screen = self.own_ship.lat < self.lat_bounds[0] or self.own_ship.lat > self.lat_bounds[1] or \
                        self.own_ship.lon < self.lon_bounds[0] or self.own_ship.lon > self.lon_bounds[1]

        if out_of_screen:
            reward -= 100.0  # Large penalty for leaving bounds
            terminated = True
            return reward, terminated, truncated, info
        else:
            # Small penalty for approaching bounds
            lat_penalty = min(0.0, (self.own_ship.lat - self.lat_bounds[0]) * (
                    self.lat_bounds[1] - self.own_ship.lat) * 0.01)
            lon_penalty = min(0.0, (self.own_ship.lon - self.lon_bounds[0]) * (
                    self.lon_bounds[1] - self.own_ship.lon) * 0.01)
            reward += lat_penalty + lon_penalty

        return reward, terminated, truncated, info

    def reset(self, seed=None, options=None) -> tuple[ObsType, dict[str, Any]]:
        # seed for random number generator
        super().reset(seed=seed)

        # reset the step counter
        self.step_counter = 0

        own_ship_data = {}
        targets_data = {}

        # TODO code the state for testing, not training, provided the target count
        if self.training_stage == 0:
            pass

        # state for wp tracking training
        elif self.training_stage == 1:  # training for wp tracking, no targets

            # placing the vessel in the center of the env
            self.own_ship.lat, self.own_ship.lon = np.mean(self.lat_bounds), np.mean(self.lon_bounds)

            # random initial speed, minimum 7 kn
            self.own_ship.speed = np.random.uniform(low=7, high=self.own_ship.max_speed)

            # fixed speed to reach wp
            self.own_ship.constant_speed = 15

            # random initial course
            self.own_ship.course = np.random.uniform(low=0, high=360)

            # place the waypoint
            while True:
                waypoint_lat = np.random.uniform(self.lat_bounds[0], self.lat_bounds[1])
                waypoint_lon = np.random.uniform(self.lon_bounds[0], self.lon_bounds[1])
                distance_to_waypoint = self.own_ship.calculate_distance((waypoint_lat, waypoint_lon))
                # Ensure waypoint is 3 NM to 20 NM away from the vessel
                if 3.0 < distance_to_waypoint < 20:
                    break

            # calculate target eta, calculated using initial speed
            target_eta = self.wp_eta

            self.waypoint.lat, self.waypoint.lon = waypoint_lat, waypoint_lon

            own_ship_data = self._generate_own_ship_data()
            own_ship_data['wp_eta'] = target_eta
            own_ship_data['wp_target_eta'] = target_eta

            targets_data = self._generate_zero_target_data(3)

        # TODO set the state for training with targets
        else:
            pass

        raw_observation = {
            "own_ship": own_ship_data,
            "targets": targets_data
        }

        self.observation = self._flatten_observation(raw_observation)

        return self.observation, {}

    def render(self):
        if self.window is None:
            # Initialize pygame
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Marine Environment")
            self.clock = pygame.time.Clock()

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        # Clear the screen
        self.window.fill((0, 0, 50))  # Dark blue background

        # Draw the own ship
        lat, lon = self.own_ship.lat, self.own_ship.lon
        course, speed = self.observation[:2]
        px, py = self._latlon_to_pixels(lat, lon)
        pygame.draw.circle(self.window, (255, 255, 255), (px, py), self.vessel_size)  # Own ship (white)

        # Draw own ship's heading line
        heading_rad = np.deg2rad(course - 90)  # Adjust course for pygame's coordinate system
        line_length = int(speed * self.scale) / 10  # Line length proportional to speed
        end_x = px + int(line_length * np.cos(heading_rad))
        end_y = py + int(line_length * np.sin(heading_rad))
        pygame.draw.line(self.window, (255, 0, 0), (px, py), (end_x, end_y), 2)  # Red heading line

        # Draw the waypoint
        waypoint_px, waypoint_py = self._latlon_to_pixels(self.waypoint.lat, self.waypoint.lon)
        pygame.draw.circle(self.window, (0, 255, 0), (waypoint_px, waypoint_py), self.vessel_size)  # Waypoint (green)

        # Draw remaining waypoints
        for wp in self.waypoints:
            waypoint_px, waypoint_py = self._latlon_to_pixels(*wp)
            pygame.draw.circle(self.window, (100, 255, 100), (waypoint_px, waypoint_py),
                               self.vessel_size)  # Remaining WPs (light green)

        # Draw the target ships
        for target in self.own_ship.detected_targets:
            # Convert target ship position to pixel coordinates
            target_px, target_py = self._latlon_to_pixels(target.lat, target.lon)
            pygame.draw.circle(self.window, (0, 0, 255), (target_px, target_py), self.vessel_size)  # Target ship (blue)

            # Draw target ship's heading line
            target_heading_rad = np.deg2rad(target.course - 90)  # Adjust course for pygame
            target_line_length = int(target.speed * self.scale)  # Line proportional to target speed
            target_end_x = target_px + int(target_line_length * np.cos(target_heading_rad))
            target_end_y = target_py + int(target_line_length * np.sin(target_heading_rad))
            pygame.draw.line(self.window, (0, 255, 255), (target_px, target_py), (target_end_x, target_end_y),
                             2)  # Cyan heading line

        # Update the display
        pygame.display.flip()
        self.clock.tick(30)  # Limit to 30 FPS

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None

    @staticmethod
    def _flatten_observation(raw_observation):
        """
        Flattens a nested dictionary with tuples of dictionaries into a 1D NumPy array.

        :param raw_observation: Dict with 'own_ship' and 'targets'
        :return: Flattened NumPy array (axis=1)
        """
        # Extract and flatten own_ship data
        own_ship_features = list(raw_observation['own_ship'].values())

        # Extract and flatten targets data
        target_features = []
        for target in raw_observation['targets']:  # Assuming `targets` is a tuple of dicts
            target_features.extend(target.values())  # Flatten each target dictionary

        # Combine all into a single NumPy array (1D)
        flat_obs = np.array(own_ship_features + target_features, dtype=np.float32)

        return flat_obs

    def _latlon_to_pixels(self, lat, lon):
        """Convert latitude and longitude to pixel coordinates."""
        # Get the map's latitude and longitude ranges
        lat_range = self.lat_bounds[1] - self.lat_bounds[0]
        lon_range = self.lon_bounds[1] - self.lon_bounds[0]
        px = int((lon - self.lon_bounds[0]) / lon_range * self.window_size)
        py = int((self.lat_bounds[1] - lat) / lat_range * self.window_size)
        return px, py

    def _generate_own_ship_data(self) -> dict[str, float]:
        result = dict()
        for key in self.OWN_SHIP_PARAMS:
            result[key] = getattr(self.own_ship, key)
        for key in self.WP_PARAMS:
            result[key] = getattr(self, key)

        return result

    def _generate_zero_target_data(self, targets_count: int) -> list[dict[str, float]]:
        result = []
        for _ in range(targets_count):
            spaces_dict = {}
            for key in self.TARGET_PARAMS:
                spaces_dict[key] = 0.0

            result.append(spaces_dict)

        return result

    def _generate_actual_target_data(self, target: 'Target') -> Dict[str, float]:
        result = dict()
        for key in self.TARGET_PARAMS:
            attr = key.removeprefix('target_')
            result[key] = getattr(target, attr)

        return result

    def _generate_observation_dict(self, observation: ObsType) -> dict[str, dict[str, Any]]:
        idx = 0
        own_ship_data = dict()
        targets_data = dict()
        for key in self.OWN_SHIP_PARAMS:
            own_ship_data[key] = observation[idx]
            idx += 1
        for key in self.WP_PARAMS:
            own_ship_data[key] = observation[idx]
            idx += 1
        for key in self.TARGET_PARAMS:
            targets_data[key] = observation[idx]
            idx += 1
        return {'own_ship': own_ship_data, 'targets': targets_data}


if __name__ == '__main__':
    env = MarineEnv(continuous=True)
    # print(env.observation_space)
    # print(flatten_space(env.observation_space))
    # print(env.observation)
    # print(env.own_ship)
    print(env.reset())
    # print(type(env.action_space))
    # print(env.step((0.7, 0.5)))
    # print(env.observation)
    target_ship = Target(position=(0.05, 0.05), course=270, speed=5, min_speed=5, max_speed=20)
    env.observation[0] = 0
    env.observation[1] = 10
    env.own_ship.lat = 0.0
    env.own_ship.lon = 0.0
    env.own_ship.course = 0.0
    env.own_ship.speed = 10.0
    env.own_ship.min_speed = 5
    env.own_ship.max_speed = 20
    env.own_ship.detected_targets = [target_ship]
    env.observation[3] = env.wp_eta
    env.observation[5] = env.observation[3]
    env.step((0, 0))
    print(env.step((0.0, 0.0)))
    print(env.step((0.0, 0)))
    print(env.step((0.0, 0)))
    print(env.step((0.0, 0)))
