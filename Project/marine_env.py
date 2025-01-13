import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from collections import deque

from utils import plane_sailing_next_position, rumbline_distance, calculate_bearing, calculate_relative_bearing


class BaseShip:
    def __init__(self, course, speed):
        self.course = course
        self.speed = speed


class OwnShip(BaseShip):
    def __init__(self, lat, lon, course, speed):
        super().__init__(course, speed)
        self.lat = lat
        self.lon = lon

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

    def distance(self, target_position):
        return rumbline_distance([self.lat, self.lon], [target_position[0], target_position[1]])

    def rel_bearing(self, target_position):
        bearing_to_waypoint = calculate_bearing(self.lat, self.lon, target_position[0], target_position[1])
        # calculate relative bearing
        return calculate_relative_bearing(self.course, bearing_to_waypoint)

    def __call__(self, *args, **kwargs):
        return [self.lat, self.lon, self.course, self.speed]


class WpMarineEnv(gym.Env):
    def __init__(self, time_scale=60, training=False):
        super().__init__()
        # Define action space (3 discrete actions)
        self.action_space = spaces.Discrete(3)
        self.training = training
        self.own_ship = OwnShip(30.25, 100.25, 23, 15)

        # Def lat and lon bounds
        self.lat_bounds = (30.0, 30.5)  # Example latitude bounds (30 NM range)
        self.lon_bounds = (100.0, 100.5)  # Example longitude bounds

        # Define observation space: [course, speed]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -180, 0]),
            high=np.array([360, 20, 180, 50]),
            dtype=np.float64
        )

        # Initialize the state
        self.state = np.zeros(4)  # [course, speed in knots, relative bearing to wp, distance to wp]
        self.time_scale = time_scale  # How much to scale simulation time (1x = real time)
        self.real_world_dt = 1 / 60.0  # Real-world time step (1 minute)

        self.sim_dt = self.real_world_dt / self.time_scale  # Scaled simulation time step

        self.total_sim_time = 0.0  # Total simulation time in hours

        # Define waypoint
        self.waypoint = np.array([30.4, 100.25])  # Example waypoint (lat, lon)
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
        # extract own ship params from the current state
        course, speed, previous_rel_wp_bearing, previous_wp_distance = self.state

        # Update based on action
        if action == 0:  # Turn port (left) by 5 degrees
            course = (course - 1) % 360
        elif action == 1:  # Turn starboard (right) by 5 degrees
            course = (course + 1) % 360
        elif action == 2:  # Keep course and speed
            pass

        # set the course
        self.own_ship.set_course(course)

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
        self.total_sim_time += self.sim_dt  # Increment simulated time

        # Reward Initialization
        reward = 0.0

        # Reward for reaching the waypoint
        if distance_to_waypoint <= self.waypoint_reach_threshold:
            reward += 100.0  # Large reward for successfully reaching the waypoint
        else:
            # Distance-Based Reward or Penalty
            distance_change = previous_wp_distance - distance_to_waypoint
            reward += max(-1.0, distance_change)  # Reward proportional to distance improvement

            # Bearing Alignment Reward
            if abs(rel_wp_bearing) <= 5:
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
            reward -= 100.0  # Large penalty for leaving bounds
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
                self.state = np.array([lat, lon, course, speed, rel_wp_bearing, distance_to_waypoint])
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
        lat = (self.lat_bounds[0] + self.lat_bounds[1]) / 2
        lon = (self.lon_bounds[0] + self.lon_bounds[1]) / 2
        self.own_ship.lat, self.own_ship.lon = lat, lon

        random_course = np.random.uniform(0, 360)  # Random course in degrees
        random_speed = np.random.uniform(5, 15)  # Random speed between 5 and 15 knots
        self.own_ship.course = random_course
        self.own_ship.speed = random_speed
        # Randomly place the waypoint, ensuring it is not too close to the vessel
        while True:
            waypoint_lat = np.random.uniform(self.lat_bounds[0], self.lat_bounds[1])
            waypoint_lon = np.random.uniform(self.lon_bounds[0], self.lon_bounds[1])
            distance_to_waypoint = rumbline_distance([lat, lon], [waypoint_lat, waypoint_lon])
            if distance_to_waypoint > 5.0 and distance_to_waypoint < 7.0:  # Ensure waypoint is 5 NM to 7 NM away from the vessel
                break
        # calculate relative bearing and distance to wp
        true_wp_bearing = calculate_bearing(lat, lon, waypoint_lat, waypoint_lon)
        relative_wp_bearing = calculate_relative_bearing(random_course, true_wp_bearing)

        # Update the state and waypoint
        self.state = np.array(
            [lat, lon, random_course, random_speed, relative_wp_bearing, distance_to_waypoint])
        self.waypoint = np.array([waypoint_lat, waypoint_lon])

        # generate additional waypoints if not in training
        if not self.training:
            for i in range(2):
                while True:
                    random_lat = np.random.uniform(self.lat_bounds[0], self.lat_bounds[1])
                    random_lon = np.random.uniform(self.lon_bounds[0], self.lon_bounds[1])
                    distance_to_waypoint = rumbline_distance([waypoint_lat, waypoint_lon], [random_lat, random_lon])
                    if distance_to_waypoint > 5.0 and distance_to_waypoint < 7.0:  # Ensure waypoint is 5 NM to 7 NM away from the last wp
                        waypoint_lat, waypoint_lon = random_lat, random_lon
                        self.waypoints.append((waypoint_lat, waypoint_lon))
                        break

                        # Reset simulation time
        self.total_sim_time = 0.0

        return self.state, {}

    def render(self, mode="human"):
        if self.window is None:
            # print("Initializing pygame...")
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Marine Environment")
            self.clock = pygame.time.Clock()
            # print("Pygame initialized.")

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        # Clear the screen
        self.window.fill((0, 0, 50))  # Dark blue background

        # Debugging state and waypoint
        # print(f"Debug: State = {self.state}, Waypoint = {self.waypoint}")

        # Draw the vessel
        lat, lon, course, speed, *rest = self.state
        px, py = self.latlon_to_pixels(lat, lon)
        # print(f"Debug: Vessel at pixels ({px}, {py})")

        pygame.draw.circle(self.window, (255, 255, 255), (px, py), self.vessel_size)

        # Draw heading as a line
        heading_rad = np.deg2rad(course - 90)  # Adjust course for pygame's coordinate system
        line_length = int(speed * self.scale)  # Line length proportional to speed
        end_x = px + int(line_length * np.cos(heading_rad))
        end_y = py + int(line_length * np.sin(heading_rad))
        # print(f"Debug: Heading line to ({end_x}, {end_y})")
        pygame.draw.line(self.window, (255, 0, 0), (px, py), (end_x, end_y), 2)

        # Draw the waypoint
        waypoint_px, waypoint_py = self.latlon_to_pixels(*self.waypoint)
        # print(f"Debug: Waypoint at pixels ({waypoint_px}, {waypoint_py})")
        pygame.draw.circle(self.window, (0, 255, 0), (waypoint_px, waypoint_py), self.vessel_size)
        # Draw the remaining wps
        for wp in self.waypoints:
            waypoint_px, waypoint_py = self.latlon_to_pixels(*wp)
            pygame.draw.circle(self.window, (100, 255, 100), (waypoint_px, waypoint_py), self.vessel_size)

        # Update the display
        pygame.display.flip()
        self.clock.tick(30)  # Limit to 30 FPS

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
