import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

#                               -------- CONFIGURATION --------
# Define colors
BLACK = (0, 0, 0)
WHITE = (1, 1, 1)
RED = (1, 0, 0)
GREEN = (0, 1, 0)
YELLOW = (1, 1, 0)
BLUE = (0, 0, 1)

# Define map dimensions and start/goal points, number of static obstacles
WIDTH = 100
HEIGHT = 150
START = (50, 20)
GOAL = (50, 120)
NUM_STATIC_OBS = 5

# Define observation radius and grid size
RADIUS = 100
SQUARE_SIZE = 10
SPEED = 1
OBSTACLE_RADIUS = SQUARE_SIZE/3

# Define initial heading angle, turn rate and number of steps
INITIAL_HEADING = 90
TURN_RATE = 5

# Define states
FREE_STATE = 0          # free space
PATH_STATE = 1          # path
COLLISION_STATE = 2     # obstacle or border
GOAL_STATE = 3          # goal point

class ASVEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    def __init__(self, render_mode = "human"):
        super(ASVEnv, self).__init__()
        self.render_mode = render_mode

        self.width = WIDTH
        self.height = HEIGHT
        self.heading = INITIAL_HEADING
        self.turn_rate = TURN_RATE
        self.speed = SPEED
        self.start = START
        self.goal = GOAL
        self.radius = RADIUS
        self.grid_size = SQUARE_SIZE
        self.center_point = (0,0)
        self.step_taken = []

        self.path = self.generate_path(self.start, self.goal)
        self.boundary = self.generate_border(self.width, self.height)
        self.goal_point = self.generate_goal(self.goal)

        # Action space and observation space
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=3, shape=(313,), dtype=np.int32)   
        # 5 possible actions: left, right, straight, accelerate, decelerate
        # 4 possible states for observation, and the shape = number of grids inside the observation radius

        self.reset()
    
    #                           -------- HELPER FUNCTIONS --------
    
    # Create a function that sets the priority of each state in case they overlap: 
    # obstacle/collision > goal point > path > free space
    def get_priority_state(self, current_state, new_state):
        if new_state == COLLISION_STATE:
            return COLLISION_STATE
        elif new_state == GOAL_STATE and current_state != COLLISION_STATE:
            return GOAL_STATE
        elif new_state == PATH_STATE and current_state not in (COLLISION_STATE, GOAL_STATE):
            return PATH_STATE
        elif current_state not in (COLLISION_STATE, GOAL_STATE, PATH_STATE):
            return FREE_STATE
        return current_state
    
    # Create a function that converts each point from the global map to a grid coordinate
    def closest_multiple(self, n, mult):
        return int((n + mult / 2) // mult) * mult
    
    # Create a function to generate a dictionary, storing the grid coordinates and state
    def fill_grid(self, objects, grid_size):
        grid_dict = {}
        for obj in objects:
            m = obj['x']
            n = obj['y']
            state = obj['state']

            m = self.closest_multiple(m, grid_size)
            n = self.closest_multiple(n, grid_size)

            if (m, n) not in grid_dict:
                grid_dict[(m, n)] = FREE_STATE
            
            grid_dict[(m, n)] = self.get_priority_state(grid_dict[(m,n)], state)
        return grid_dict

    # Create a function that generate grid coordinates (x,y) from global map
    def generate_grid(self, radius, square_size, center):
        x = np.arange(-radius + square_size, radius, square_size)
        y = np.arange(-radius + square_size, radius, square_size)
        grid = []
        for i in x:
            for j in y:
                if np.sqrt(i ** 2 + j ** 2) <= radius:
                    grid.append((center[0] + i, center[1] + j))
        return grid
    
    #                           -------- MAP GENERATION --------

    # Create a function to generate borders around the map
    def generate_border(self, map_width, map_height):
        boundary = []
        for x in range(0, map_width + 1):
            boundary.append({'x': x, 'y': 0, 'state': COLLISION_STATE})             # lower boundary
            boundary.append({'x': x, 'y': map_height, 'state': COLLISION_STATE})    # upper boundary
        for y in range(0, map_height + 1):
            boundary.append({'x': 0, 'y': y, 'state': COLLISION_STATE})             # left boundary
            boundary.append({'x': map_width, 'y': y, 'state': COLLISION_STATE})     # right boundary
        return boundary
    
    # Create a function to generate static obstacles
    def generate_static_obstacles(self, num_obs, map_width, map_height):
        obstacles = []
        # Generate random obstacles around the map
        for _ in range(num_obs):
            x = np.random.randint(0, map_width)
            y = np.random.randint(0, map_height)
            obstacles.append({'x': x, 'y': y, 'state': COLLISION_STATE})
        # Generate 2 random obstacles along the path
        for _ in range(2):
            x = self.start[0]
            y = np.random.randint(self.start[1] + 20, self.goal[1] - 20)
            obstacles.append({'x': x, 'y': y, 'state': COLLISION_STATE})
        return obstacles
        # x = 50
        # y = 70
        # obstacles.append({'x': x, 'y': y, 'state': COLLISION_STATE})
        # return obstacles
    
    # Function that generate a path line (list/array of points)
    def generate_path(self, start_point, goal_point):
        path = []
        num_points = goal_point[1] - start_point[1]     # straight vertical line
        for i in range(num_points):
            y = start_point[1] + i
            path.append({'x': start_point[0], 'y': y, 'state': PATH_STATE})
        return path
    
    def generate_goal(self, goal_point):
        goal = []
        goal.append({'x': goal_point[0], 'y': goal_point[1], 'state': GOAL_STATE})
        return goal

    def reset(self, seed=None, options=None):
        # super().reset(seed=seed)
        self.obstacles = self.generate_static_obstacles(3, self.width, self.height)
        self.objects_environment = self.obstacles + self.path + self.boundary + self.goal_point
        self.grid_dict = self.fill_grid(self.objects_environment, self.grid_size)

        self.step_count = 0
        self.current_heading = self.heading
        self.current_speed = self.speed
        self.position = self.start
        self.done = False
        self.grid = self.generate_grid(self.radius, self.grid_size, self.position)
        return self.get_observation(), {}
    
    def get_observation(self):
        current_pos = self.position
        new_grid = self.generate_grid(self.radius, self.grid_size, current_pos)
        observation = np.zeros(len(new_grid), dtype = np.int32)
        for idx, (x, y) in enumerate(new_grid):
            state = self.grid_dict.get((self.closest_multiple(x, self.grid_size), self.closest_multiple(y, self.grid_size)), FREE_STATE)
            observation[idx] = state
        return observation
    
    def step(self, action):
        if action == 0:     # go straight
            self.position = (self.position[0] + self.speed * np.cos(np.radians(self.current_heading)),
                        self.position[1] + self.speed * np.sin(np.radians(self.current_heading)))
        elif action == 1:   # turn left
            self.current_heading += self.turn_rate
            self.position = (self.position[0] + self.speed * np.cos(np.radians(self.current_heading)),
                        self.position[1] + self.speed * np.sin(np.radians(self.current_heading)))
        elif action == 2:   # turn right
            self.current_heading -= self.turn_rate
            self.position = (self.position[0] + self.speed * np.cos(np.radians(self.current_heading)),
                        self.position[1] + self.speed * np.sin(np.radians(self.current_heading)))
        elif action == 3:   # accelerate
            self.current_speed += 0.5
            if self.current_speed > 2:
                self.current_speed = 2
            self.position = (self.position[0] + self.speed * np.cos(np.radians(self.current_heading)),
                            self.position[1] + self.speed * np.sin(np.radians(self.current_heading)))
        elif action == 4:   # decelerate
            self.current_speed -= 0.5
            if self.current_speed < 1:
                self.current_speed = 1
            self.position = (self.position[0] + self.speed * np.cos(np.radians(self.current_heading)),
                            self.position[1] + self.speed * np.sin(np.radians(self.current_heading)))

        self.step_count += 1
        self.step_taken.append((self.position[0], self.position[1]))
        reward = self.calculate_reward(self.position)
        terminated = self.check_done(self.position)
        observation = self.get_observation()
        
        return observation, reward, terminated, False, {}
    
    def calculate_reward(self, position):
        x, y = position
        state = self.grid_dict.get((self.closest_multiple(x, self.grid_size), self.closest_multiple(y, self.grid_size)), FREE_STATE)
        distance_to_path = self.calculate_distance_to_path(self.position)
        distance_to_goal = self.calculate_distance_to_goal(self.position)
        # Calculate the distance to the nearest obstacle
        nearest_obstacle_distance = self.calculate_distance_to_nearest_obstacle(self.position)
        
        # Set a threshold distance for significant penalty
        danger_zone_threshold = self.grid_size * 2  
        
        reward = 0
        if state == COLLISION_STATE:
            reward -= 1000
        elif state == GOAL_STATE:
            reward += 500
        elif state == PATH_STATE:
            reward += 10 + 1 / (distance_to_goal + 1)
        elif state == FREE_STATE:
            reward -= (1 + distance_to_path + distance_to_goal*0.01)
        
        # Add a penalty for being too close to an obstacle
        # if nearest_obstacle_distance < danger_zone_threshold:
        #     reward -= 500 * (1 - nearest_obstacle_distance / danger_zone_threshold)
        if nearest_obstacle_distance < danger_zone_threshold:
            reward -= 10 / (nearest_obstacle_distance + 1)

        # Add a reward for getting closer to the goal
        reward += 1 / (distance_to_goal + 1)

        return reward
    
    def calculate_distance_to_path(self, position):
        path_x = [point['x'] for point in self.path]
        path_y = [point['y'] for point in self.path]
        min_distance = float('inf')
        for px, py in zip(path_x, path_y):
            distance = np.sqrt((position[0] - px) ** 2 + (position[1] - py) ** 2)
            if distance < min_distance:
                min_distance = distance
        return min_distance
    
    def calculate_distance_to_goal(self, position):
        goal_x = [point['x'] for point in self.goal_point]
        goal_y = [point['y'] for point in self.goal_point]
        min_distance = float('inf')
        for px, py in zip(goal_x, goal_y):
            distance = np.sqrt((position[0] - px) ** 2 + (position[1] - py) ** 2)
            if distance < min_distance:
                min_distance = distance
        return min_distance
    
    def calculate_distance_to_nearest_obstacle(self, position):
        x, y = position
        min_distance = float('inf')
        
        for (obstacle_x, obstacle_y), state in self.grid_dict.items():
            if state == COLLISION_STATE:
                distance = np.sqrt((x - obstacle_x)**2 + (y - obstacle_y)**2)
                if distance < min_distance:
                    min_distance = distance
        
        return min_distance
    
    def check_done(self, position):        
        x, y = position
        state = self.grid_dict.get((self.closest_multiple(x, self.grid_size), self.closest_multiple(y, self.grid_size)), FREE_STATE)
        
        # If the agent collide with obstacles or boundary
        if state == COLLISION_STATE:
            return True
        # If the agent reached goal
        elif state == GOAL_STATE:
            return True
        # If the total number of steps are 250 or above
        elif self.step_count >= 500:
            return True
        return False

    def render(self, mode="human"):
        if mode == 'human':
            if not hasattr(self, 'fig'):
                self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 8))

                self.ax1.set_aspect('equal')
                self.ax1.set_title('MAP')
                self.ax1.set_xlim(-self.radius, self.width + self.radius)
                self.ax1.set_ylim(-self.radius, self.height + self.radius)

                self.ax2.set_aspect('equal')
                self.ax2.set_title('OBSERVATION')
                self.ax2.set_xlim(-self.radius, self.radius)
                self.ax2.set_ylim(-self.radius, self.radius)

                self.agent_1, = self.ax1.plot([], [], marker='^', color=BLUE)
                self.agent_2, = self.ax2.plot([], [], marker='^', color=BLUE)
                self.observation_horizon1 = plt.Circle(self.start, self.radius, color=BLUE, fill=False)
                self.observation_horizon2 = plt.Circle((0, 0), self.radius, color=BLUE, fill=False)
                self.ax1.add_patch(self.observation_horizon1)
                self.ax2.add_patch(self.observation_horizon2)

                self.ax1.plot(self.start[0], self.start[1], marker='o', color=BLUE)
                self.ax1.plot(self.goal[0], self.goal[1], marker='o', color=YELLOW)

                for obj in self.boundary:
                    boundary_line = plt.Rectangle((obj['x'], obj['y']), 1, 1, edgecolor=BLACK, facecolor=BLACK)
                    self.ax1.add_patch(boundary_line)
                
                path_x = [point['x'] for point in self.path]
                path_y = [point['y'] for point in self.path]
                self.ax1.plot(path_x, path_y, '-', color=GREEN)

                for obj in self.obstacles:
                    self.ax1.plot(obj['x'], obj['y'], marker='o', color=RED)

            self.agent_1.set_data(self.position[0], self.position[1])
            self.agent_1.set_marker((3, 0, self.current_heading - 90))

            self.observation_horizon1.center = self.position

            new_grid = self.generate_grid(self.radius, self.grid_size, self.position)
            for rect in getattr(self, 'grid_patches', []):
                rect.remove()
            self.grid_patches = []

            for (cx, cy) in new_grid:
                state = self.grid_dict.get((self.closest_multiple(cx, self.grid_size), self.closest_multiple(cy, self.grid_size)), FREE_STATE)
                color = WHITE
                if state == COLLISION_STATE:
                    color = RED
                elif state == PATH_STATE:
                    color = GREEN
                elif state == GOAL_STATE:
                    color = YELLOW
                rect = plt.Rectangle((cx - self.grid_size / 2 - self.position[0], cy - self.grid_size / 2 - self.position[1]), self.grid_size, self.grid_size,
                                    edgecolor='gray', facecolor=color)
                rect.set_zorder(1)
                self.grid_patches.append(rect)
                self.ax2.add_patch(rect)

            self.agent_2.set_data(0, 0)
            self.agent_2.set_marker((3, 0, self.current_heading - 90))
            self.agent_2.set_zorder(3)

            self.observation_horizon2.center = (0, 0)
            self.observation_horizon2.set_zorder(2)

            plt.draw()
            plt.pause(0.01)

# Test the environment with random actions
if __name__ == '__main__':
    env = ASVEnv()
    obs = env.reset()

    print("Observation Space Shape", env.observation_space.shape)
    print("Sample observation", env.observation_space.sample())

    print("Action Space Shape", env.action_space.n)
    print("Action Space Sample", env.action_space.sample())

    for _ in range(500):  # Run for 100 steps or until done
        action = env.action_space.sample()  # Take a random action
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done:
            break

    env.close()
