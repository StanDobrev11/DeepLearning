import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from collections import deque

from utils import plane_sailing_next_position, plane_sailing_position, rumbline_distance, calculate_bearing, \
    calculate_relative_bearing


class BaseShip:
    def __init__(self, course, speed):
        self.course = course
        self.speed = speed


class TargetShip(BaseShip):
    def __init__(self, lat, lon, course, speed):
        super().__init__(course, speed)
        self.lat = lat
        self.lon = lon
        self.relative_course = None
        self.relative_speed = None
        self.relative_bearing = None
        self.relative_distance = None
        self.cpa = None
        self.tcpa = None
        self.stand_on = True  # Defines if the vessel should give way or is stand on

    def update_relative_position(self, own_ship):
        """
        Update the relative bearing and distance to the own ship.
        """
        self.relative_bearing = calculate_bearing(self.lat, self.lon, own_ship.lat, own_ship.lon)
        self.relative_distance = rumbline_distance([self.lat, self.lon], [own_ship.lat, own_ship.lon])

    def calculate_tcpa(self, own_ship):
        distance = rumbline_distance([self.lat, self.lon],
                                     [own_ship.lat, own_ship.lon]
                                     )
        return distance / self.relative_speed

    def is_dangerous(self, cpa_threshold=1.0, tcpa_threshold=15.0):
        """
        Determine if the target ship is dangerous based on CPA and TCPA thresholds.
        :param cpa_threshold: CPA distance below which a target is considered dangerous
        :param tcpa_threshold: TCPA time below which a target is considered dangerous
        :return: True if dangerous, False otherwise
        """
        return self.cpa <= cpa_threshold and 0 <= self.tcpa <= tcpa_threshold


class OwnShip(BaseShip):
    def __init__(self, lat, lon, course, speed):
        super().__init__(course, speed)
        self.lat = lat
        self.lon = lon
        self.detected_ships = []  # List of detected TargetShip objects

    def distance(self, target_position):
        return rumbline_distance([self.lat, self.lon], [target_position[0], target_position[1]])

    def rel_bearing(self, target_position):
        true_bearing = calculate_bearing(self.lat, self.lon, target_position[0], target_position[1])
        # calculate relative bearing
        return calculate_relative_bearing(self.course, true_bearing)

    def __call__(self, *args, **kwargs):
        return [self.lat, self.lon, self.course, self.speed]

    def set_course(self, course_change):
        self.course += course_change

    def next_position(self, time_interval):
        self.lat, self.lon = plane_sailing_next_position(
            [self.lat, self.lon],
            self.course,
            self.speed,
            time_interval
        )
        return self.lat, self.lon

    def detect_target_ship(self, target_ship):
        """
        Detect and update the relative position and CPA/TCPA of a target ship.
        """
        target_ship.update_relative_position(self)
        target_ship.calculate_cpa_tcpa(self, time_interval=1)  # Use 1-minute interval for extrapolation
        self.detected_ships.append(target_ship)

    def get_dangerous_ships(self, cpa_threshold=1.0, tcpa_threshold=15.0):
        """
        Identify and return a list of dangerous ships.
        """
        dangerous_ships = [
            ship for ship in self.detected_ships if ship.is_dangerous(cpa_threshold, tcpa_threshold)
        ]
        return dangerous_ships

    def calculate_cpa_tcpa(self, target_ship: TargetShip):
        """
        Calculate CPA and TCPA between own ship and target ship.

        :param own_speed: Speed of the own ship (knots).
        :param own_course: Course of the own ship (degrees).
        :param target_speed: Speed of the target ship (knots).
        :param target_course: Course of the target ship (degrees).
        :param rel_position: Relative position of the target ship [Rx, Ry] (nautical miles).
        :return: (CPA, TCPA)
        """
        own_speed = self.speed
        target_speed = target_ship.speed
        # Convert angles to radians
        own_course_rad = np.radians(self.course)
        target_course_rad = np.radians(target_ship.course)

        # Own ship velocity components
        own_vx = own_speed * np.sin(own_course_rad)
        own_vy = own_speed * np.cos(own_course_rad)

        # Target ship velocity components
        target_vx = target_speed * np.sin(target_course_rad)
        target_vy = target_speed * np.cos(target_course_rad)

        # Relative velocity components
        rel_vx = target_vx - own_vx
        rel_vy = target_vy - own_vy

        # Relative position components
        own_lat_rad = np.radians(self.lat)

        # Relative position components
        rx = (target_ship.lon - self.lon) * 60 * np.cos(own_lat_rad)  # Longitudinal distance (NM)
        ry = (target_ship.lat - self.lat) * 60  # Latitudinal distance (NM)

        # Relative velocity magnitude squared
        v_rel_squared = rel_vx ** 2 + rel_vy ** 2

        # Dot product of relative position and velocity
        r_dot_v_rel = rx * rel_vx + ry * rel_vy

        # Calculate TCPA
        if v_rel_squared != 0:
            tcpa = -r_dot_v_rel / v_rel_squared
        else:
            tcpa = float('inf')  # No relative motion, TCPA is infinite

        # Calculate CPA
        cpa_squared = rx ** 2 + ry ** 2 - (
                r_dot_v_rel ** 2 / v_rel_squared) if v_rel_squared != 0 else rx ** 2 + ry ** 2
        cpa = np.sqrt(max(0, cpa_squared))  # Ensure CPA is non-negative

        return cpa, tcpa


class MarineEnv(gym.Env):
    def __init__(self, time_scale=1, training=False):
        super().__init__()
        self.training = training  # if in training, reduced number of wps

        # Define action space (3 discrete actions)
        self.action_space = spaces.Discrete(3)

        # Define own ship and list, containing target ships
        self.own_ship = OwnShip(None, None, None, None)  # will be initialized in the reset() method
        self.target_ship = TargetShip(None, None, None, None)
        self.target_ships = []

        # Def lat and lon bounds
        self.lat_bounds = (30.0, 30.5)  # Example latitude bounds (30 NM range)
        self.lon_bounds = (100.0, 100.5)  # Example longitude bounds

        # Define observation space: [course, speed, rel wp bearing, wp distance]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -180, 0]),
            high=np.array([360, 20, 180, 50]),
            dtype=np.float64
        )

        # Initialize the state
        self.state = np.zeros(4)  # [course, speed in knots, relative bearing to wp, distance to wp]
        self.time_scale = time_scale  # How much to scale simulation time (1x = real time)
        self.real_world_dt = 1 / 60.0  # Real-world time step (1 minute)

        self.sim_dt = self.real_world_dt * self.time_scale  # Scaled simulation time step

        self.total_sim_time = 0.0  # Total simulation time in hours

        # Define waypoint
        self.waypoint = np.zeros(2)  # waypoint (lat, lon) will be defined in the reset() method
        self.waypoint_reach_threshold = 0.003  # Limit considered reaching of waypoint
        self.waypoints = deque([])

        # Pygame setup
        self.window_size = 600  # Pixels for visualization
        self.scale = self.window_size / ((self.lat_bounds[1] - self.lat_bounds[0]) * 60)  # Pixels per NM
        self.window = None
        self.clock = None
        self.vessel_size = 5  # Vessel radius in pixels

    def latlon_to_pixels(self, lat, lon):
        """Convert latitude and longitude to pixel coordinates."""
        # Get the map's latitude and longitude ranges
        lat_range = self.lat_bounds[1] - self.lat_bounds[0]
        lon_range = self.lon_bounds[1] - self.lon_bounds[0]
        px = int((lon - self.lon_bounds[0]) / lon_range * self.window_size)
        py = int((self.lat_bounds[1] - lat) / lat_range * self.window_size)
        return px, py

    def step(self, action):
        # move the target ship
        for target in self.target_ships:
            lat, lon = plane_sailing_next_position(
                [target.lat, target.lon],
                target.course,
                target.speed,
                time_interval=self.sim_dt
            )
            target.lat, target.lon = lat, lon
            target.cpa, target.tcpa = self.own_ship.calculate_cpa_tcpa(target)

        # extract own ship params from the current state
        course, speed, previous_rel_wp_bearing, previous_wp_distance = self.state

        # Update based on action
        if action == 0:  # Turn port (left) by 1 degree
            course = (course - 1) % 360
        elif action == 1:  # Turn starboard (right) by 1 degree
            course = (course + 1) % 360
        elif action == 2:  # Keep course
            pass

        # set the course
        self.own_ship.course = course

        lat, lon = self.own_ship.next_position(time_interval=self.sim_dt)
        # Clip latitude and longitude to bounds
        lat = np.clip(lat, self.lat_bounds[0], self.lat_bounds[1])
        lon = np.clip(lon, self.lon_bounds[0], self.lon_bounds[1])
        self.own_ship.lat, self.own_ship.lon = lat, lon

        # Calculate distance to waypoint
        distance_to_waypoint = self.own_ship.distance([self.waypoint[0], self.waypoint[1]])
        # Calculate relative bearing to wp
        rel_wp_bearing = self.own_ship.rel_bearing([self.waypoint[0], self.waypoint[1]])

        # Update state
        self.state = np.array([course, speed, rel_wp_bearing, distance_to_waypoint])
        self.total_sim_time += self.sim_dt / self.time_scale  # Increment simulated time

        # Reward Initialization
        reward = 0.0

        # Reward for reaching the waypoint
        if distance_to_waypoint <= self.waypoint_reach_threshold:
            reward += 1000.0  # Large reward for successfully reaching the waypoint
        else:
            # Distance-Based Reward or Penalty
            distance_change = previous_wp_distance - distance_to_waypoint
            # reward += max(-1.0, distance_change * 10)  # Reward proportional to distance improvement
            reward += distance_change * 10 if distance_change > 0 else -1

            # Bearing Alignment Reward
            if abs(rel_wp_bearing) <= 1:
                reward += 5.0
            elif abs(rel_wp_bearing) <= 5:
                reward += 2.0  # Strong reward for near-perfect alignment
            elif abs(rel_wp_bearing) <= 15:
                reward += 1.0  # Moderate reward for good alignment
            elif abs(rel_wp_bearing) <= 30:
                reward += 0.5  # Small reward for decent alignment
            else:
                reward -= 0.5  # Penalty for poor alignment

            # Reward for Improving Bearing Alignment
            bearing_change = abs(previous_rel_wp_bearing) - abs(rel_wp_bearing)
            reward += max(-0.25, bearing_change * 0.25)  # Small reward for improving alignment

        # Penalty for Going Out of Bounds
        out_of_screen = lat <= self.lat_bounds[0] or lat >= self.lat_bounds[1] or \
                        lon <= self.lon_bounds[0] or lon >= self.lon_bounds[1]

        if out_of_screen:
            reward -= 1000.0  # Large penalty for leaving bounds
        else:
            # Small penalty for approaching bounds
            lat_penalty = min(0.0, (lat - self.lat_bounds[0]) * (self.lat_bounds[1] - lat) * 0.01)
            lon_penalty = min(0.0, (lon - self.lon_bounds[0]) * (self.lon_bounds[1] - lon) * 0.01)
            reward += lat_penalty + lon_penalty

        # Episode Termination Conditions
        waypoint_reached = distance_to_waypoint <= self.waypoint_reach_threshold
        if waypoint_reached:
            try:
                self.waypoint[0], self.waypoint[1] = self.waypoints.popleft()
                distance_to_waypoint = rumbline_distance([lat, lon], [self.waypoint[0], self.waypoint[1]])
                bearing_to_waypoint = calculate_bearing(lat, lon, self.waypoint[0], self.waypoint[1])
                rel_wp_bearing = calculate_relative_bearing(course, bearing_to_waypoint)
                self.state = np.array([course, speed, rel_wp_bearing, distance_to_waypoint])
                waypoint_reached = False
            except:
                pass
        done = waypoint_reached or out_of_screen

        return self.state, reward, done, waypoint_reached, {"total_sim_time": self.total_sim_time}

    def reset(self, seed=None, options=None):
        # Handle the seed for random number generation
        if seed is not None:
            np.random.seed(seed)

        # Randomly place the vessel within the bounds
        # random_lat = np.random.uniform(self.lat_bounds[0], self.lat_bounds[1])
        # random_lon = np.random.uniform(self.lon_bounds[0], self.lon_bounds[1])

        # fix the vessel in the middle of the observation space
        # lat = (self.lat_bounds[0] + self.lat_bounds[1]) / 2
        # lon = (self.lon_bounds[0] + self.lon_bounds[1]) / 2

        # fix the vessel in the lower left corner
        lat, lon = 30.05, 100.05

        self.own_ship.lat, self.own_ship.lon = lat, lon

        # random_course = np.random.uniform(0, 360)  # Random course in degrees
        # self.own_ship.course = random_course

        random_speed = np.random.uniform(5, 15)  # Random speed between 5 and 15 knots
        self.own_ship.speed = random_speed
        # Randomly place the waypoint, ensuring it is not too close to the vessel
        while True:
            waypoint_lat = np.random.uniform(self.lat_bounds[0], self.lat_bounds[1])
            waypoint_lon = np.random.uniform(self.lon_bounds[0], self.lon_bounds[1])
            distance_to_waypoint = rumbline_distance([lat, lon], [waypoint_lat, waypoint_lon])
            if distance_to_waypoint > 15.0 and distance_to_waypoint < 20:  # Ensure waypoint is 5 NM to 7 NM away from the vessel
                break
        # calculate relative bearing and distance to wp
        true_wp_bearing = calculate_bearing(lat, lon, waypoint_lat, waypoint_lon)
        own_course = true_wp_bearing
        self.own_ship.course = own_course

        relative_wp_bearing = calculate_relative_bearing(own_course, true_wp_bearing)

        self.target_ship = self.place_dangerous_target_ship(scene='head-on')

        # Update the state and waypoint
        self.state = np.array(
            [own_course, random_speed, relative_wp_bearing, distance_to_waypoint])
        self.waypoint = np.array([waypoint_lat, waypoint_lon])

        # generate additional waypoints if not in training
        if not self.training:
            for i in range(2):
                while True:
                    random_lat = np.random.uniform(self.lat_bounds[0], self.lat_bounds[1])
                    random_lon = np.random.uniform(self.lon_bounds[0], self.lon_bounds[1])
                    distance_to_waypoint = rumbline_distance([waypoint_lat, waypoint_lon], [random_lat, random_lon])
                    if distance_to_waypoint > 25.0 and distance_to_waypoint < 35.0:  # Ensure waypoint is 5 NM to 7 NM away from the last wp
                        waypoint_lat, waypoint_lon = random_lat, random_lon
                        self.waypoints.append((waypoint_lat, waypoint_lon))
                        break

                        # Reset simulation time
        self.total_sim_time = 0.0

        return self.state, {}

    def calculate_target_course_and_speed(self, own_ship: OwnShip, rel_vector: [float, float]):
        own_course_rad, own_speed = np.radians(own_ship.course), own_ship.speed
        rel_course_rad, rel_speed = np.radians(rel_vector[0]), rel_vector[1]

        # Own ship velocity components (in Cartesian coordinates)
        own_vx = own_speed * np.sin(own_course_rad)
        own_vy = own_speed * np.cos(own_course_rad)

        # Relative velocity components (in Cartesian coordinates)
        rel_vx = rel_speed * np.sin(rel_course_rad)
        rel_vy = rel_speed * np.cos(rel_course_rad)

        # Target ship absolute velocity components (vector addition)
        target_vx = own_vx + rel_vx
        target_vy = own_vy + rel_vy

        # Calculate target ship speed (magnitude of the velocity vector)
        target_speed = np.sqrt(target_vx ** 2 + target_vy ** 2)

        # Calculate target ship course (angle of the velocity vector)
        target_course = (np.degrees(np.arctan2(target_vx, target_vy)) + 360) % 360

        return target_course, target_speed

    def place_dangerous_target_ship(self, scene: str):
        """
        Place a TargetShip on a potentially dangerous track with specified CPA and TCPA.

        :param scene: Defines the situation, i.e static, head-on, overtaking, crossing. The scene will define the status of the target ship as stand-on or give-way in further training
        :param cpa_target: Desired closest point of approach (in NM).
        :param tcpa_target: Desired time to closest point of approach (in minutes).
        :return: TargetShip object added to the environment.
        """
        # Own ship's parameters
        own_lat, own_lon = self.own_ship.lat, self.own_ship.lon
        own_course = self.own_ship.course
        own_speed = self.own_ship.speed

        # Target ship

        # define the position of the target vessel to comply with ColReg
        relative_bearing = None
        relative_speed = None
        if scene == 'head-on' or 'static':
            relative_bearing = np.random.uniform(-5, 5)  # Relative bearing in degrees
            relative_speed = np.random.uniform(5, 15) + own_speed
        elif scene == 'crossing' and self.target_ship.stand_on:
            relative_bearing = np.random.uniform(5, 112.5)
            relative_speed = np.random.uniform(2, 15 + own_speed)
        elif scene == 'crossing' and not self.target_ship.stand_on:
            relative_bearing = np.random.uniform(-5, -112.5)
            relative_speed = np.random.uniform(2, 15 + own_speed)

        # if the relative course == 180 + relative bearing, the target is on a collision course.
        # to generate relative course != to 180 + relative bearing bss CPA, we need distance or TCPA
        # to calculate TCPA we need relative speed
        initial_distance = np.random.uniform(5, 7)
        cpa = np.random.random_sample() * 2 - 1  # random CPA between -1 and 1 NM
        tcpa = initial_distance / relative_speed
        # calculating relative course in degrees
        deviation = np.degrees(np.arctan2(cpa, initial_distance))
        true_bearing = (own_course + relative_bearing) % 360
        reversed_true_bearing = (true_bearing + 180) % 360
        true_course = reversed_true_bearing + deviation

        target_course, target_speed = self.calculate_target_course_and_speed(
            self.own_ship, [true_course, relative_speed]
        )

        # Calculate the initial position of the TargetShip
        target_lat, target_lon = plane_sailing_position(
            [own_lat, own_lon], true_bearing, initial_distance
        )

        #  TargetShip and add it to the environment
        self.target_ship.lat = target_lat
        self.target_ship.lon = target_lon
        self.target_ship.course = target_course
        self.target_ship.speed = target_speed
        self.target_ship.relative_bearing = relative_bearing
        self.target_ship.relative_speed = relative_speed
        self.target_ship.cpa = cpa
        self.target_ship.tcpa = tcpa

        self.target_ships.append(self.target_ship)

        return self.target_ship

    def render(self, mode="human"):
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
        course, speed, *rest = self.state
        px, py = self.latlon_to_pixels(lat, lon)
        pygame.draw.circle(self.window, (255, 255, 255), (px, py), self.vessel_size)  # Own ship (white)

        # Draw own ship's heading line
        heading_rad = np.deg2rad(course - 90)  # Adjust course for pygame's coordinate system
        line_length = int(speed * self.scale)  # Line length proportional to speed
        end_x = px + int(line_length * np.cos(heading_rad))
        end_y = py + int(line_length * np.sin(heading_rad))
        pygame.draw.line(self.window, (255, 0, 0), (px, py), (end_x, end_y), 2)  # Red heading line

        # Draw the waypoint
        waypoint_px, waypoint_py = self.latlon_to_pixels(*self.waypoint)
        pygame.draw.circle(self.window, (0, 255, 0), (waypoint_px, waypoint_py), self.vessel_size)  # Waypoint (green)

        # Draw remaining waypoints
        for wp in self.waypoints:
            waypoint_px, waypoint_py = self.latlon_to_pixels(*wp)
            pygame.draw.circle(self.window, (100, 255, 100), (waypoint_px, waypoint_py),
                               self.vessel_size)  # Remaining WPs (light green)

        # Draw the target ships
        for target_ship in self.target_ships:
            # Convert target ship position to pixel coordinates
            target_px, target_py = self.latlon_to_pixels(target_ship.lat, target_ship.lon)
            pygame.draw.circle(self.window, (0, 0, 255), (target_px, target_py), self.vessel_size)  # Target ship (blue)

            # Draw target ship's heading line
            target_heading_rad = np.deg2rad(target_ship.course - 90)  # Adjust course for pygame
            target_line_length = int(target_ship.speed * self.scale)  # Line proportional to target speed
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


if __name__ == '__main__':
    env = MarineEnv()
    # print()
    # Reset the environment [course, speed, wp rel bearing, wp distance]
    state, info = env.reset()

    for t in range(10000):  # Maximum steps for visualization
        action = env.action_space.sample()
        # env.place_dangerous_target_ship('head-on')
        # Take the action in the environment
        next_state, reward, terminated, truncated, info = env.step(action)

        # Render the environment
        # env.render()

        # Prepare the next state
        if terminated:
            break
    print(f"Simulated Time: {info['total_sim_time']} hours")
    env.close()
