import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class MarineEnv(gym.Env):
    def __init__(self, time_scale=60):
        super(MarineEnv, self).__init__()

        # Define action space (5 discrete actions)
        self.action_space = spaces.Discrete(5)

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

        # Pygame setup
        self.window_size = 600  # Pixels for visualization
        self.scale = self.window_size / ((self.lat_bounds[1] - self.lat_bounds[0]) * 60)  # Pixels per NM
        self.window = None
        self.clock = None
        self.vessel_size = 5  # Vessel radius in pixels

    
    def latlon_to_pixels(self, lat, lon):
        """Convert latitude and longitude to pixel coordinates."""
        lat_range = self.lat_bounds[1] - self.lat_bounds[0]
        lon_range = self.lon_bounds[1] - self.lon_bounds[0]
        px = int((lon - self.lon_bounds[0]) / lon_range * self.window_size)
        py = int((self.lat_bounds[1] - lat) / lat_range * self.window_size)
        return px, py

    def step(self, action):
        lat, lon, course, speed = self.state

        # Update based on action
        if action == 0:  # Turn port (left) by 5 degrees
            pass
            # course = (course - 5) % 360
        elif action == 1:  # Turn starboard (right) by 5 degrees
            pass
            # course = (course + 5) % 360
        elif action == 2:  # Slow down by 1 knot
            pass
            # speed = max(speed - 1, 0)  # Min speed is 0 knots
        elif action == 3:  # Speed up by 1 knot
            pass
            # speed = min(speed + 1, 20)  # Max speed is 20 knots
        elif action == 4:  # Keep course and speed
            pass

        # Update position using real-world scaling
        rad = np.deg2rad(course)
        d_lat = (speed * self.sim_dt) / 60.0  # Degrees latitude per simulated time step
        d_lon = (speed * self.sim_dt) / 60.0 * np.cos(np.deg2rad(lat))  # Degrees longitude

        lon += d_lon * np.cos(rad)
        lat += d_lat * np.sin(rad)

        # Clip latitude and longitude to bounds
        lat = np.clip(lat, self.lat_bounds[0], self.lat_bounds[1])
        lon = np.clip(lon, self.lon_bounds[0], self.lon_bounds[1])

        # Update state
        self.state = np.array([lat, lon, course, speed])
        self.total_sim_time += self.sim_dt  # Increment simulated time

        # Define reward (can be customized later)
        reward = 0.0

        # The episode ends when speed is 0 or vessel moves out of bounds
        done = (speed == 0) or (lat <= self.lat_bounds[0] or lat >= self.lat_bounds[1] or
                                lon <= self.lon_bounds[0] or lon >= self.lon_bounds[1])

        return self.state, reward, done, False, {"total_sim_time": self.total_sim_time}

    def reset(self, seed=None, options=None):
        # Handle the seed for random number generation
        if seed is not None:
            np.random.seed(seed)

        # Reset the state
        self.state = np.array([30.1, 100.1, 45.0, 15.0])  # Example fixed state
        self.total_sim_time = 0.0  # Reset total simulation time
        return self.state, {}

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
        heading_rad = np.deg2rad(-course)  # Adjust course for pygame's coordinate system
        line_length = int(speed * self.scale)  # Line length proportional to speed
        end_x = px + int(line_length * np.cos(heading_rad))
        end_y = py + int(line_length * np.sin(heading_rad))
        pygame.draw.line(self.window, (255, 0, 0), (px, py), (end_x, end_y), 2)

        # Update the display
        pygame.display.flip()
        self.clock.tick(30)  # Limit to 30 FPS

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
