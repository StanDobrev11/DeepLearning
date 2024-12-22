import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from utils import plane_sailing_next_position, rumbline_distance, calculate_bearing


class MarineEnv(gym.Env):
    def __init__(self, time_scale=60):
        super(MarineEnv, self).__init__()

        # Define action space (5 discrete actions)
        self.action_space = spaces.Discrete(3)

        # Define observation space: [lat, lon, course, speed]
        self.lat_bounds = (30.0, 30.5)  # Example latitude bounds (30 NM range)
        self.lon_bounds = (100.0, 100.5)  # Example longitude bounds
        self.observation_space = spaces.Box(
            low=np.array([self.lat_bounds[0], self.lon_bounds[0], 0, 0]),
            high=np.array([self.lat_bounds[1], self.lon_bounds[1], 360, 20]),
            dtype=np.float64
        )

        # Initialize the state
        self.state = np.array([30.1, 100.1, 45.0, 15.0])  # [lat, lon, course, speed in knots]
        self.time_scale = time_scale  # How much to scale simulation time (1x = real time)
        self.real_world_dt = 1 / 60.0  # Real-world time step (1 minute)

        self.sim_dt = self.real_world_dt / self.time_scale  # Scaled simulation time step

        self.total_sim_time = 0.0  # Total simulation time in hours

        # Define waypoint
        self.waypoint = np.array([30.4, 100.25])  # Example waypoint (lat, lon)
        self.waypoint_reach_treshold = 0.01  # Limit considered reaching of waypoint

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

    def calculate_distance_to_waypoint(self, lat, lon):
        """Calculate the distance to the waypoint in nautical miles."""
        delta_lat = (self.waypoint[0] - lat) * 60  # Convert degrees to NM
        delta_lon = (self.waypoint[1] - lon) * 60 * np.cos(np.radians(lat))  # Adjust for latitude
        return np.sqrt(delta_lat**2 + delta_lon**2)  # Pythagorean theorem

    def step(self, action):
        lat, lon, course, speed = self.state
        previous_distance_to_waypoint = self.calculate_distance_to_waypoint(lat, lon)

        # Update based on action
        if action == 0:  # Turn port (left) by 5 degrees
            course = (course - 1) % 360
        elif action == 1:  # Turn starboard (right) by 5 degrees
            course = (course + 1) % 360
        # elif action == 2:  # Slow down by 1 knot
        #     speed = max(speed - 1, 5)  # Min speed is 0 knots
        # elif action == 3:  # Speed up by 1 knot
        #     speed = min(speed + 1, 20)  # Max speed is 20 knots
        elif action == 2:  # Keep course and speed
            pass

        # Update position using the plane sailing method
        lat, lon = plane_sailing_next_position([lat, lon], course, speed, time_interval=self.sim_dt)

        # Clip latitude and longitude to bounds
        lat = np.clip(lat, self.lat_bounds[0], self.lat_bounds[1])
        lon = np.clip(lon, self.lon_bounds[0], self.lon_bounds[1])

        # Update state
        self.state = np.array([lat, lon, course, speed])
        self.total_sim_time += self.sim_dt  # Increment simulated time

        # Calculate distance to waypoint
        distance_to_waypoint = self.calculate_distance_to_waypoint(lat, lon)

        # Calculate bearing to the waypoint
        bearing_to_waypoint = calculate_bearing(lat, lon, self.waypoint[0], self.waypoint[1])

        # Reward Logic
        reward = 0.0

        # Reward for reaching the waypoint
        if distance_to_waypoint <= self.waypoint_reach_treshold:
            reward += 100.0  # Large positive reward for reaching the waypoint
        else:
            # Distance-based reward/penalty
            if distance_to_waypoint < previous_distance_to_waypoint:
                reward += 1.0  # Small positive reward for decreasing distance
            else:
                reward -= 1.0  # Small penalty for increasing distance

            # Heading alignment reward/penalty
            heading_diff = abs(course - bearing_to_waypoint)
            heading_diff = min(heading_diff, 360 - heading_diff)  # Smallest angle difference

            if heading_diff <= 10:  # Within 10° of bearing
                reward += 1.0  # Reward for alignment
            else:
                reward -= heading_diff / 10.0  # Penalty proportional to misalignment

        # The episode ends when speed is 0, vessel moves out of bounds, or waypoint is reached
        waypoint_reached = distance_to_waypoint <= self.waypoint_reach_treshold  # Within 0.1 NM
        done = waypoint_reached or speed == 0 or \
               lat <= self.lat_bounds[0] or lat >= self.lat_bounds[1] or \
               lon <= self.lon_bounds[0] or lon >= self.lon_bounds[1]

        return self.state, reward, done, waypoint_reached, {"total_sim_time": self.total_sim_time}

    def reset(self, seed=None, options=None):
        # Handle the seed for random number generation
        if seed is not None:
            np.random.seed(seed)

        # Randomly place the vessel within the bounds
        random_lat = np.random.uniform(self.lat_bounds[0], self.lat_bounds[1])
        random_lon = np.random.uniform(self.lon_bounds[0], self.lon_bounds[1])
        random_course = np.random.uniform(0, 360)  # Random course in degrees
        random_speed = np.random.uniform(5, 15)  # Random speed between 5 and 15 knots

        # Randomly place the waypoint, ensuring it is not too close to the vessel
        while True:
            waypoint_lat = np.random.uniform(self.lat_bounds[0], self.lat_bounds[1])
            waypoint_lon = np.random.uniform(self.lon_bounds[0], self.lon_bounds[1])
            distance_to_waypoint = rumbline_distance([random_lat, random_lon], [waypoint_lat, waypoint_lon])
            if distance_to_waypoint > 5.0:  # Ensure waypoint is at least 5 NM away from the vessel
                break

        # Update the state and waypoint
        self.state = np.array([random_lat, random_lon, random_course, random_speed])
        self.waypoint = np.array([waypoint_lat, waypoint_lon])

        # Reset simulation time
        self.total_sim_time = 0.0

        return self.state, {}

    def normalize_state(self, state):
        # Normalize latitude, longitude, and speed
        normalized_lat = (state[0] - self.lat_bounds[0]) / (self.lat_bounds[1] - self.lat_bounds[0])
        normalized_lon = (state[1] - self.lon_bounds[0]) / (self.lon_bounds[1] - self.lon_bounds[0])
        normalized_speed = state[3] / 20.0  # Max speed is 20 knots
        return np.array([normalized_lat, normalized_lon, state[2] / 360.0, normalized_speed])

    def render(self, mode="human"):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Marine Environment")
            self.clock = pygame.time.Clock()

        # Clear the screen
        self.window.fill((0, 0, 50))  # Dark blue background

        # Draw the vessel
        lat, lon, course, speed = self.state
        px, py = self.latlon_to_pixels(lat, lon)
        pygame.draw.circle(self.window, (255, 255, 255), (px, py), self.vessel_size)

        # Draw heading as a line
        heading_rad = np.deg2rad(course - 90)  # Adjust course for pygame's coordinate system
        line_length = int(speed * self.scale)  # Line length proportional to speed
        end_x = px + int(line_length * np.cos(heading_rad))
        end_y = py + int(line_length * np.sin(heading_rad))
        pygame.draw.line(self.window, (255, 0, 0), (px, py), (end_x, end_y), 2)

        # Draw the waypoint
        waypoint_px, waypoint_py = self.latlon_to_pixels(*self.waypoint)
        pygame.draw.circle(self.window, (0, 255, 0), (waypoint_px, waypoint_py), self.vessel_size)

        # Update the display
        pygame.display.flip()
        self.clock.tick(30)  # Limit to 30 FPS

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None


