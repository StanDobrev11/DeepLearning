import random
from typing import Tuple, Optional, Any, Dict, List

import gymnasium as gym
import pygame

from gymnasium import spaces
from gymnasium.core import ObsType
from gymnasium.spaces.utils import flatten, flatten_space

import numpy as np

from utils import plane_sailing_position
from vessels import OwnShip, Target, StaticObject


class MarineEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    OWN_SHIP_PARAMS: List[str] = []
    WP_PARAMS: List[str] = []
    TARGET_PARAMS: List[str] = []

    def __init__(
            self,
            # environment properties
            initial_lat: float = 0.0,
            initial_lon: float = 0.0,
            env_range: int = 15,  # defines the size of the field
            timescale: int = 1,  # defines the step size, defaults to 1 min step
            continuous: bool = False,

            max_turn_angle: int = 20,  # rate of turn defaults to 20 deg per min
            max_speed_change: float = 0.5,  # rate of speed change knots / min
            waypoint_reach_threshold: float = 0.2,
            render_mode=None,
            # target properties
            cpa_threshold: float = 1.0,  # in nautical miles
            tcpa_threshold: float = 12,  # in minutes
            # the limits at witch the own ship is no longer considered "stand-on" and must act to avoid collision
            cpa_limit: float = 0.1,
            tcpa_limit: float = 3,
            training_stage: int = 1,  # defines different training stages, 0 for no training
            total_targets: Optional[int] = None
    ):
        super(MarineEnv, self).__init__()

        self.step_counter = 0
        self.training_stage = training_stage
        self.total_targets = total_targets

        # target props
        self.cpa_threshold = cpa_threshold  # minimum safe distance
        self.tcpa_threshold = tcpa_threshold
        self.cpa_limit = cpa_limit  # limit to act if own ship is stand-on
        self.tcpa_limit = tcpa_limit

        # initialize the environment bounds
        self.env_range = env_range
        self.lat_bounds: Tuple[float, float] = (initial_lat, initial_lat + env_range / 60)
        self.lon_bounds: Tuple[float, float] = (initial_lon, initial_lon + env_range / 60)

        # define the timescale and scaling of real word time to simulation time
        self.timescale = timescale

        # initialize action space
        if continuous:
            # the agent can continuously adjust course and speed
            self.max_turn_angle = max_turn_angle * timescale  # scaling the turn
            self.max_speed_change = max_speed_change * timescale  # scaling the speed change
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

        self.window_size = 800  # Pixels for visualization
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
        if self.own_ship.speed > 0:
            self.own_ship.speed = min(self.own_ship.max_speed, self.own_ship.speed)
        else:
            self.own_ship.speed = max(self.own_ship.min_speed, self.own_ship.speed)

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
                if target.distance <= 3:  # define dangerous distance targets when agent must act
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

        def wp_following_reward(rwrd: float):

            # Distance-Based Reward or Penalty
            previous_wp_distance = previous_data['own_ship']['wp_distance']
            current_wp_distance = current_data['own_ship']['wp_distance']

            distance_change = previous_wp_distance - current_wp_distance
            rwrd += max(-1.0, distance_change * 10)  # Reward proportional to distance improvement

            # bearing alignment reward
            current_wp_relative_bearing = current_data['own_ship']['wp_relative_bearing']
            if abs(current_wp_relative_bearing) <= 1:
                rwrd += 7.0
            elif abs(current_wp_relative_bearing) <= 5:
                rwrd += 2.0  # Strong reward for near-perfect alignment
            elif abs(current_wp_relative_bearing) <= 15:
                rwrd += 1.0  # Moderate reward for good alignment
            elif abs(current_wp_relative_bearing) <= 30:
                rwrd += 0.5  # Small reward for decent alignment
            else:
                rwrd -= 0.5  # Penalty for poor alignment

            # penalize small unnecessary course changes
            if abs(course_change) < 1:
                rwrd += 1.0  # Reward for keeping steady course
            elif abs(course_change) > 5:
                rwrd -= 0.5  # Small penalty for medium course change
            elif abs(course_change) > 8:
                rwrd -= 0.8  # Penalize for large course changes

            return rwrd

        previous_data = self._generate_observation_dict(previous_obs)
        current_data = self._generate_observation_dict(current_obs)

        # terminated - the episode ends because the agent has reached a goal or violated an environment rule
        # (e.g., collision, reaching the waypoint, ect).
        # truncated - the episode ends due to external constraints like a maximum time step limit,
        # regardless of whether the agent reached a goal or not.
        terminated, truncated = False, False
        info = {'total_steps': self.step_counter}
        reward = 0.0

        previous_course = previous_data['own_ship']['course']
        current_course = current_data['own_ship']['course']
        course_change = current_course - previous_course

        # reaching the wp -> large reward and episode termination
        if self.wp_distance < self.waypoint_reach_threshold:
            return 100.0, True, False, info

        # reward for wp tracking, training stage 1
        if self.training_stage == 1:
            reward = wp_following_reward(reward)

        # TODO reward for training stage 2, one target
        if self.training_stage == 2:
            reward = wp_following_reward(reward)

            # iterate over each target and modify the reward
            for i, target in enumerate(self.own_ship.dangerous_targets):
                # huge penalty for collision
                if target.distance < self.cpa_limit:
                    reward -= 100
                    terminated = True

                # penalty for CPA
                if target.cpa < abs(self.cpa_threshold):
                    reward -= 1 % i if i > 0 else 1  # for less dangerous targets, less penalty
                else:
                    reward += 1 % i if i > 0 else 1

                # penalize alteration to port for head-on and crossing
                if current_data['targets']['target_relative_bearing'] > 0:
                    if course_change < 0:
                        reward -= 0.5
                    else:
                        reward += 0.5

                # remove target if no longer dangerous
                if target.tcpa < 0:
                    target.is_dangerous = False
                    self.own_ship.dangerous_targets.remove(target)

                # reintroducing waypoint tracking with lower weight
                previous_wp_distance = previous_data['own_ship']['wp_distance']
                current_wp_distance = current_data['own_ship']['wp_distance']
                distance_change = previous_wp_distance - current_wp_distance

                reward += max(-0.5, distance_change * 5)  # ✅ Lower weight for WP tracking

        # TODO reward for training stage 3, two targets
        # TODO reward for training stage 4, three targets

        # reward for keeping ETA steady
        current_eta = current_data['own_ship']['wp_eta']
        target_eta = current_data['own_ship']['wp_target_eta']

        if -2 <= current_eta - target_eta <= 2:
            reward += 0.1
        else:
            reward -= 1

        # if target_eta < -6:
        #     truncated = True
        #     return reward, terminated, truncated, info

        # Penalty for Going Out of Bounds
        out_of_screen = self.own_ship.lat <= self.lat_bounds[0] or self.own_ship.lat >= self.lat_bounds[1] or \
                        self.own_ship.lon <= self.lon_bounds[0] or self.own_ship.lon >= self.lon_bounds[1]

        if out_of_screen:
            reward -= 100.0  # Large penalty for leaving bounds
            terminated = True
            return reward, terminated, truncated, info

        return reward, terminated, truncated, info

    def reset(self, seed=None, options=None) -> tuple[ObsType, dict[str, Any]]:
        # seed for random number generator
        super().reset(seed=seed)

        def place_waypoint(min_range: int, max_range: int) -> tuple[float, float]:
            while True:
                waypoint_lat = np.random.uniform(self.lat_bounds[0], self.lat_bounds[1])
                waypoint_lon = np.random.uniform(self.lon_bounds[0], self.lon_bounds[1])
                distance_to_waypoint = self.own_ship.calculate_distance((waypoint_lat, waypoint_lon))
                # Ensure waypoint is at correct distance from the vessel
                if min_range < distance_to_waypoint < max_range:
                    return waypoint_lat, waypoint_lon

        # reset the step counter
        self.step_counter = 0

        own_ship_data = {}
        targets_data = {}

        # random initial speed, minimum 7 kn
        self.own_ship.speed = np.random.uniform(low=7, high=self.own_ship.max_speed)

        # TODO code the state for testing, not training, provided the target count
        if self.training_stage == 0:
            pass

        # state for wp tracking training
        elif self.training_stage == 1:  # training for wp tracking, no targets

            # placing the vessel in the center of the env
            self.own_ship.lat, self.own_ship.lon = np.mean(self.lat_bounds), np.mean(self.lon_bounds)

            # random initial course
            self.own_ship.course = np.random.uniform(low=0, high=360)

            # place the waypoint
            self.waypoint.lat, self.waypoint.lon = place_waypoint(3, 20)

            # calculate target eta, calculated using initial speed
            target_eta = self.wp_eta

            own_ship_data = self._generate_own_ship_data()
            own_ship_data['wp_eta'] = target_eta
            own_ship_data['wp_target_eta'] = target_eta

            targets_data = self._generate_zero_target_data(3)

        # set the state for training with 1 target
        elif self.training_stage == 2:

            # set own position at random corner
            delta_lat = (self.lon_bounds[1] - self.lon_bounds[0]) / 6
            delta_lon = (self.lat_bounds[1] - self.lat_bounds[0]) / 6

            # Define corners
            corners = [
                (self.lat_bounds[0] + delta_lat, self.lon_bounds[0] + delta_lon),  # Lower Left
                (self.lat_bounds[0] + delta_lat, self.lon_bounds[1] - delta_lon),  # Lower Right
                (self.lat_bounds[1] - delta_lat, self.lon_bounds[0] + delta_lon),  # Top Left
                (self.lat_bounds[1] - delta_lat, self.lon_bounds[1] - delta_lon)  # Top Right
            ]

            # Select a random corner
            self.own_ship.lat, self.own_ship.lon = random.choice(corners)

            # place the waypoint
            self.waypoint.lat, self.waypoint.lon = place_waypoint(7, 12)

            target_eta = self.wp_eta
            own_ship_data = self._generate_own_ship_data()
            # course to match wp + minor deviation
            self.own_ship.course = self.own_ship.calculate_true_bearing(self.waypoint) + random.uniform(-5, 5)
            own_ship_data['course'] = self.own_ship.course
            # update the relative bearing
            own_ship_data['wp_relative_bearing'] = self.own_ship.calculate_relative_bearing(self.waypoint)
            own_ship_data['wp_eta'] = target_eta
            own_ship_data['wp_target_eta'] = target_eta

            target = self._place_dangerous_target_ship()
            target_data = self._generate_actual_target_data(target)

            targets_data = [target_data] + self._generate_zero_target_data(self.training_stage)

            # add target to detected targets
            self.own_ship.detected_targets.append(target)

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
        line_length = int(speed * self.scale) / 6  # Line length proportional to speed
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
            target_line_length = int(target.speed * self.scale) / 6  # Line proportional to target speed
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

    def _place_dangerous_target_ship(self, scene: Optional[str] = None) -> 'Target':
        """
        Place a TargetShip on a potentially dangerous track with specified CPA and TCPA.

        :param scene: Defines the situation, i.e static, head-on, overtaking, crossing. The scene will define the status of the target ship as stand-on or give-way in further training
        :param cpa_target: Desired closest point of approach (in NM).
        :param tcpa_target: Desired time to closest point of approach (in minutes).
        :return: TargetShip object added to the environment.
        """

        def calculate_target_course_and_speed(rel_course: float, rel_speed: float) -> tuple[float, float]:
            course_rad, speed = np.radians(self.own_ship.course), self.own_ship.speed
            rel_course_rad, rel_speed = np.radians(rel_course), rel_speed

            # Own ship velocity components (in Cartesian coordinates)
            own_vx = speed * np.sin(course_rad)
            own_vy = speed * np.cos(course_rad)

            # Relative velocity components (in Cartesian coordinates)
            rel_vx = rel_speed * np.sin(rel_course_rad)
            rel_vy = rel_speed * np.cos(rel_course_rad)

            # Target ship absolute velocity components (vector addition)
            target_vx = own_vx + rel_vx
            target_vy = own_vy + rel_vy

            # Calculate target ship speed (magnitude of the velocity vector)
            tgt_speed = np.sqrt(target_vx ** 2 + target_vy ** 2)

            # Calculate target ship course (angle of the velocity vector)
            tgt_course = (np.degrees(np.arctan2(target_vx, target_vy)) + 360) % 360

            return tgt_course, tgt_speed

        scenes = ['head-on', 'overtaking', 'crossing', 'static']

        if scene is None:
            scene = random.choice(scenes)

        # Own ship's parameters
        own_lat, own_lon = self.own_ship.lat, self.own_ship.lon
        own_course = self.own_ship.course
        own_speed = self.own_ship.speed

        # Target ship
        # if the relative course == 180 + relative bearing, the target is on a collision course.
        # to generate relative course != to 180 + relative bearing bss CPA, we need distance or TCPA
        # to calculate TCPA we need relative speed
        initial_distance = np.random.uniform(3, 6)
        cpa = np.random.random_sample() * 2 - 1  # random CPA between -1 and 1 NM

        # deviation angle for randomness
        deviation_angle = np.degrees(np.arctan2(cpa, initial_distance))

        # define the position of the target vessel to comply with ColReg
        relative_bearing = None
        relative_speed = None
        if scene == 'head-on':
            relative_bearing = np.random.uniform(-5, 5)  # Relative bearing in degrees
            relative_speed = np.random.uniform(5, 15) + own_speed
        elif scene == 'static':
            relative_bearing = np.random.uniform(-5, 5)
            relative_speed = own_speed + np.random.sample()
        # own ship is give way vessel
        elif scene == 'crossing':
            relative_bearing = np.random.uniform(5, 117.5)
            relative_speed = np.random.uniform(2, 15 + own_speed)
        # elif scene == 'crossing':  # own ship stands on
        #     relative_bearing = np.random.uniform(-5, -117.5)
        #     relative_speed = np.random.uniform(2, 15 + own_speed)
        elif scene == 'overtaking':
            relative_bearing = np.random.uniform(-45, 45)
            relative_speed = np.random.uniform(2, own_speed * 0.9)

        # print('Scene: ', scene)

        # calculating relative course in degrees and add deviation
        true_target_bearing = (own_course + relative_bearing) % 360
        reversed_true_target_bearing = (true_target_bearing + 180) % 360
        relative_course = reversed_true_target_bearing + deviation_angle

        target_course, target_speed = calculate_target_course_and_speed(
            relative_course, relative_speed
        )
        # Calculate the initial position of the TargetShip
        target_lat, target_lon = plane_sailing_position(
            [own_lat, own_lon], true_target_bearing, initial_distance
        )

        #  create target and add it to the environment
        target = Target(
            position=(target_lat, target_lon),
            course=target_course,
            speed=target_speed,
        )
        self.own_ship.update_target(target)

        return target

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
    env = MarineEnv(continuous=True, training_stage=2)
    # print(env.observation_space)
    # print(flatten_space(env.observation_space))
    # print(env.observation)
    # print(env.own_ship)
    # print(env.reset())
    # print(type(env.action_space))
    # print(env.step((0.7, 0.5)))
    # print(env.observation)
    # target_ship = Target(position=(0.05, 0.05), course=270, speed=5, min_speed=5, max_speed=20)
    # env.observation[0] = 0
    # env.observation[1] = 10
    # env.own_ship.lat = 0.0
    # env.own_ship.lon = 0.0
    # env.own_ship.course = 0.0
    # env.own_ship.speed = 10.0
    # env.own_ship.min_speed = 5
    # env.own_ship.max_speed = 20
    # env.own_ship.detected_targets = [target_ship]
    # env.observation[3] = env.wp_eta
    # env.observation[5] = env.observation[3]
    # env.step((0, 0))
    # print(env.step((0.0, 0.0)))
    # print(env.step((0.0, 0)))
    # print(env.step((0.0, 0)))
    # print(env.step((0.0, 0)))
    print(env.reset())
    total_reward = 0
    for _ in range(1000):
        state, reward, terminated, truncated, info = env.step((0, 0))
        total_reward += reward

        print(state)
        print(reward)
        print(total_reward)

        if terminated or truncated:
            print(info)
            break

    print('Total reward: {}'.format(total_reward))
