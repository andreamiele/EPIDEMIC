import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from scipy.signal import convolve2d


S = 0
I = 1
R = 2
V = 3
DISCOUNTED_INFECTION_IMPACT_SCORE = 5
def create_observation_space(grid_size, window_size):
    # Create bounds for the grid window values
    disease_bounds = {
        'low': np.zeros((window_size * window_size,), dtype=np.int32),
        'high': np.full((window_size * window_size,), 3, dtype=np.int32)
    }

    # Create bounds for the agent's location
    location_bounds = {
        'low': np.zeros((2,), dtype=np.int32),
        'high': np.full((2,), grid_size - 1, dtype=np.int32)
    }

    # Concatenate bounds for the complete observation space
    obs_bounds = {
        'low': np.concatenate([disease_bounds['low'], location_bounds['low']]),
        'high': np.concatenate([disease_bounds['high'], location_bounds['high']])
    }

    # Define the observation space
    observation_space = spaces.Box(low=obs_bounds['low'], high=obs_bounds['high'], dtype=np.int32)
    
    return observation_space

# Usage

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, max_iterations = 200, size=10, init_infect = 1, window_size=3,num_infect=1, infect_prob = 0.5, recov_prob = 0.1, num_states = 4, days = 3, max_vaccine=20,n_vaccine=20, decay_steps=60000, decay_amount=5, random=False):
        
        self.size = size 
        window_size = size  # Grid size
        self.size = size 
        self.grid_size = size  # Grid size
        self.window_size = window_size  # Observation window size
        self.half_window = window_size // 2
        self.previous_infection_impact_scores=None
        self.current_infection_impact_scores=None
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.agent_position = None
        self.grid = None
        self.next_grid = None
        self.vac_saved = 0
        self.wrong_vaccination = 0
        self.trajectory = []
        self.location_revisited = 0
        self.num_infect = num_infect
        self.state_counts = {}
        
        self.infect_prob = infect_prob
        self.num_states = num_states
        self.recov_prob = recov_prob
        self.vac_saved = 0
        self.days=days
        self.channels = None
        self.initial_infect = None
        self.n_vaccine = n_vaccine
        self.max_vaccine= max_vaccine
        self.decay_steps = decay_steps  # New parameter for step decay interval
        self.decay_amount = decay_amount  # Initial number of vaccines
        self.episode_counter = 0
        self.random=random
        self.observation_space = create_observation_space(self.size, self.window_size)
        #self.observation_space = self.create_observation_space()
        self.action_space = spaces.Discrete(5)

        self.action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
    def create_observation_space(self):
        # Define the bounds for the observation window
        window_size = self.window_size * self.window_size
        obs_low = np.zeros(window_size, dtype=np.int32)
        obs_high = np.full(window_size, 3, dtype=np.int32)
        return spaces.Box(low=obs_low, high=obs_high, dtype=np.int32)
    def reset(self, seed=None, options=None):
        self.previous_infection_impact_scores = self.current_infection_impact_scores
        episodes_passed = self.episode_counter // self.decay_steps
        self.max_vaccine = max(0, 40 - episodes_passed * self.decay_amount)
        
        self.n_vaccine = self.max_vaccine
        #Needed to seed self.np_random
        super().reset(seed=seed)

        #Choose the agent's location uniformly at random
        self.trajectory = []
        self.agent_position = self.np_random.integers(0, self.size, size=2, dtype=int)
        self.trajectory.append(self.agent_position)

        # Create an initial grid of all S
        grid = np.full((self.size, self.size), S, dtype=int)

        # Calculate the total number of cells
        total_cells = self.size * self.size

        if self.random:
            I_indices = []
        
            # Place the first infected cell randomly
            first_index = np.random.choice(self.size * self.size)
            first_x, first_y = divmod(first_index, self.size)
            grid[first_x, first_y] = I
            I_indices.append((first_x, first_y))

            while len(I_indices) < self.num_infect:
                # Select a random infected cell from the existing ones
                base_x, base_y = I_indices[np.random.choice(len(I_indices))]
                neighbors = self.get_neighbors(np.array([base_x, base_y]))
                
                # Filter out neighbors that are already infected or out of bounds
                valid_neighbors = [(n[0], n[1]) for n in neighbors if grid[n[0], n[1]] == S]

                if valid_neighbors:
                    next_x, next_y = valid_neighbors[np.random.choice(len(valid_neighbors))]
                    grid[next_x, next_y] = I
                    I_indices.append((next_x, next_y))
                else:
                    # In case no valid neighbors are found (shouldn't happen with reasonable num_infect)
                    break

            self.initial_infect = [x * self.size + y for x, y in I_indices]
        else:
            grid[5,5]=I
            self.initial_infect = (5,5)
        #Initialize the grid with some disease states --> only 0 and 1
        self.grid = grid

        #observation = self.observations().astype(np.int32)  # Cast to int32
        #observation = self.observations2()
        observation = self.get_centered_observation().astype(np.int32)
        info = self.info()

        #Reset variables
        
        self.location_revisited= 0
        self.wrong_vaccination = 0
        self.current_iteration = 0

        if self.render_mode == "human":
            self.render_frame()
        self.episode_counter += 1

        return observation, info
    
    def get_observation_window(self):
        # Get the indices of the window around the agent
        x, y = self.agent_position
        x_min = max(0, x - self.half_window)
        x_max = min(self.grid_size, x + self.half_window + 1)
        y_min = max(0, y - self.half_window)
        y_max = min(self.grid_size, y + self.half_window + 1)

        window = self.grid[x_min:x_max, y_min:y_max]

        # Pad the window if it's at the border of the grid
        if window.shape[0] < self.window_size or window.shape[1] < self.window_size:
            padded_window = np.full((self.window_size, self.window_size), S, dtype=int)
            padded_window[:window.shape[0], :window.shape[1]] = window
            return padded_window.flatten()
        return window.flatten()

    def observations2(self):
        return self.get_observation_window()
    
    def get_neighbors_distance_2(self, position):
        neighbors = []
        directions = [
            np.array([2, 0]),  # right
            np.array([0, 2]),  # down
            np.array([-2, 0]),  # left
            np.array([0, -2]),  # up
            np.array([2, 2]),  # down-right
            np.array([-2, 2]),  # down-left
            np.array([2, -2]),  # up-right
            np.array([-2, -2]),  # up-left
            np.array([2, 1]),  # right-down
            np.array([2, -1]),  # right-up
            np.array([-2, 1]),  # left-down
            np.array([-2, -1]),  # left-up
            np.array([1, 2]),  # down-right
            np.array([-1, 2]),  # down-left
            np.array([1, -2]),  # up-right
            np.array([-1, -2])  # up-left
        ]
        for direction in directions:
            neighbor = np.clip(position + direction, 0, self.size - 1)
            neighbors.append(neighbor)
        return neighbors
        
    def get_neighbors(self, position):
        neighbors = []
        directions = [
            np.array([1, 0]),  # right
            np.array([0, 1]),  # down
            np.array([-1, 0]),  # left
            np.array([0, -1]),  # up
            np.array([1, 1]),  # down-right
            np.array([-1, 1]),  # down-left
            np.array([1, -1]),  # up-right
            np.array([-1, -1])  # up-left
        ]
        for direction in directions:
            neighbor = np.clip(position + direction, 0, self.size - 1)
            neighbors.append(neighbor)
        return neighbors
    
    def step(self, action):
        terminated = False
        #Keep track of iterations
        self.current_iteration+= 1
        intermediate_reward = 0
        #Update the agent location if the agent moves
        if action <= V :
            # Map the action (element of {0,1,2,3}) to the direction we walk in
            direction = self.action_to_direction[action]
            # We use `np.clip` to make sure we don't leave the grid
            self.agent_position = np.clip(
                self.agent_position + direction, 0, self.size - 1
            )
            if any(np.array_equal(self.agent_position, loc) for loc in self.trajectory):
                self.location_revisited += 1
                intermediate_reward -= 3  # Penalize revisiting cells
            else:
                intermediate_reward += 1  # Reward for visiting new cells
            self.trajectory.append(self.agent_position)
            # Provide small reward for moving closer to infected individuals
            neighbors = self.get_neighbors(self.agent_position)
            if any(self.grid[neighbor[0], neighbor[1]] == I for neighbor in neighbors):
                intermediate_reward += 5  # Small reward for moving closer to infected individuals
            # Additional reward for being near infected individuals at distance 2
            neighbors_2 = self.get_neighbors_distance_2(self.agent_position)
            if any(self.grid[neighbor[0], neighbor[1]] == I for neighbor in neighbors_2):
                intermediate_reward += 2  # Small reward for being near infected individuals at distance 2
        else:
            x, y = self.agent_position
            if self.grid[self.agent_position[0], self.agent_position[1]] == S:
                neighbors = self.get_neighbors(self.agent_position)
                if any(self.grid[neighbor[0], neighbor[1]] == I for neighbor in neighbors):
                    intermediate_reward += 50  # Small reward for moving closer to infected individuals
                else:
                    intermediate_reward -= 15  # Penalize vaccinating away from infected individuals
                if any(self.grid[neighbor[0], neighbor[1]] == V for neighbor in neighbors):
                    intermediate_reward += 20  # Small reward for vaccinating near to a vaccination
                neighbors2 = self.get_neighbors_distance_2(self.agent_position)
                if any(self.grid[neighbor[0], neighbor[1]] == I for neighbor in neighbors2):
                    intermediate_reward += 15  # Small reward for moving closer to infected individuals  
            if self.grid[self.agent_position[0], self.agent_position[1]] == I:
                    intermediate_reward += 15
            self.grid_update()
            self.n_vaccine-=1
        # Determine if the episode should terminate
        num_infected = np.sum(self.grid == I)
        if self.n_vaccine == 0:
            terminated=True
        if num_infected == 0:
            terminated=True
        # An episode is done iff the agent has reached the target or max number of iterations
        truncated = self.current_iteration >= self.max_iterations
        
        #future_infected_count = self.look_ahead_simulation(steps=3)
        #look_ahead_reward = -future_infected_count/100 *5
        
        
        #Compute disease propagation
        if truncated or terminated:
            self.update_grid()
            

        if truncated or terminated:
            final_reward = self.reward()
            #final_reward = self.simple_reward()
        else:
            final_reward = 0
        
        # Combine intermediate reward with final reward
        #reward = intermediate_reward + final_reward # more complex reward function
        #reward = final_reward + look_ahead_reward   
        reward = final_reward # simpler reward function

        #Update observation
        #observation = self.observations().astype(np.int32)
        #observation = self.observations2()
        observation = self.get_centered_observation().astype(np.int32)
        if self.render_mode == "human":
            self.render_frame()
        
        #Update info
        info = self.info()
        return observation, reward, terminated , truncated, info 

    
    def get_centered_observation(self):
        # Find the position of the infected cell(s)
        infected_positions = np.argwhere(self.grid == I)
        if infected_positions.size == 0:
            center_position = self.agent_position
        else:
            center_position = infected_positions[0]

        x, y = center_position
        half_window = self.half_window

        # Get the indices of the window around the center position
        x_min = max(0, x - half_window)
        x_max = min(self.size, x + half_window + 1)
        y_min = max(0, y - half_window)
        y_max = min(self.size, y + half_window + 1)

        window = self.grid[x_min:x_max, y_min:y_max]

        # Pad the window if it's at the border of the grid
        if window.shape[0] < self.window_size or window.shape[1] < self.window_size:
            padded_window = np.full((self.window_size, self.window_size), S, dtype=int)
            padded_window[:window.shape[0], :window.shape[1]] = window
            observation = np.concatenate([padded_window.flatten(), self.agent_position])
            return observation
        observation = np.concatenate([window.flatten(), self.agent_position])
        return observation
    
    def clone_state(self):
        """Clone the current state of the environment."""
        return {
            'grid': self.grid.copy(),
            'agent_position': self.agent_position.copy(),
            'n_vaccine': self.n_vaccine,
            'current_iteration': self.current_iteration
        }

    def restore_state(self, state):
        """Restore the environment to a previous state."""
        self.grid = state['grid'].copy()
        self.agent_position = state['agent_position'].copy()
        self.n_vaccine = state['n_vaccine']
        self.current_iteration = state['current_iteration']

    def simulate_disease_propagation(self, steps):
        """Simulate disease propagation for a given number of steps."""
        grid_size = self.grid.shape[0]
        channels_matrix = np.zeros((self.num_states, grid_size, grid_size), dtype=int)
        for state_index in range(self.num_states):
            channels_matrix[state_index, :, :] = (self.grid == state_index).astype(int)
        
        kernel = np.ones((3, 3))
        kernel[1, 1] = 0

        for _ in range(steps):
            I_neighbors = convolve2d(channels_matrix[I], kernel, mode='same', boundary='fill', fillvalue=0)
            transmission_prob = 1 - (1 - self.infect_prob) ** I_neighbors
            new_infections = (np.random.rand(*transmission_prob.shape) < transmission_prob).astype(int)
            new_infections = new_infections * (1 - channels_matrix[V])
            random_mask = np.random.rand(*channels_matrix[R].shape)
            recovery_mask = random_mask < self.recov_prob
            channels_matrix[R] += ((new_infections == 1) & recovery_mask).astype(int)
            new_infections = new_infections * (1 - channels_matrix[R])
            channels_matrix[I] = np.clip(new_infections + channels_matrix[I] - channels_matrix[R], 0, 1)
            channels_matrix[S] = channels_matrix[S] - (channels_matrix[S] * channels_matrix[I])
            self.grid = np.argmax(channels_matrix, axis=0)

    def look_ahead_simulation(self, steps=3):
        """Perform a look-ahead simulation to estimate future infection count."""
        original_state = self.clone_state()
        
        # Simulate disease propagation for the specified number of steps
        self.simulate_disease_propagation(steps)
            
        future_infected_count = np.sum(self.grid == I)
        
        # Restore the original state
        self.restore_state(original_state)
        
        return future_infected_count

    
    
    def render(self):
        if self.render_mode == "rgb_array":
             return self.render_frame()
        return
    
    
    def update_grid(self):
        grid_size = self.grid.shape[0]
        channels_matrix = np.zeros((self.num_states, grid_size, grid_size), dtype=int)
        for state_index in range(self.num_states):
            channels_matrix[state_index, :, :] = (self.grid == state_index).astype(int)
        self.channels= channels_matrix
        kernel = np.ones((3, 3))
        kernel[1, 1] = 0
        
        for i in range(3):
            I_neighbors = convolve2d(self.channels[1], kernel, mode='same', boundary='fill', fillvalue=0)
            transmission_prob = 1 - (1 - self.infect_prob) ** I_neighbors      
            new_infections = (np.random.rand(*transmission_prob.shape) < transmission_prob).astype(int)
            self.vac_saved = np.sum(((self.channels[3] == 1) & (self.channels[1] == 0) & (new_infections == 1)).astype(int))
            new_infections = new_infections * (1 - self.channels[3])
            random_mask = np.random.rand(*self.channels[2].shape)
            recovery_mask = random_mask < self.recov_prob
            self.channels[2] += ((new_infections == 1) & recovery_mask).astype(int)
            new_infections = new_infections * (1 - self.channels[2])
            self.channels[1] = np.clip(new_infections + self.channels[1] - self.channels[2], 0, 1)         
            self.channels[0] = self.channels[0] - (self.channels[0] * self.channels[1])
            self.grid=np.argmax(self.channels, axis=0)
            if self.render_mode == "human":
                self.render_frame()
        self.next_grid = np.argmax(self.channels, axis=0)
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def seed(self, seed= None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return 
    
    def grid_update(self):
        x,y = self.agent_position
        self.grid[x,y] = 3        
        return 

    def observations(self):
        return np.concatenate((self.grid.flatten(), self.agent_position))
    
    def info(self):
        unique, counts = np.unique(self.next_grid, return_counts=True)
        counts = counts / (self.size ** 2)
        state_counts = dict(zip(unique, counts))
        for state in [S, I, R, V]:
            state_counts.setdefault(state, 0)
        self.state_counts = state_counts
        info = {
            'state_counts': self.state_counts,
            'final_grid': self.grid.copy(),  # Ensure this is set correctly
            'initial_infect': self.initial_infect,
            'num_vaccine': self.max_vaccine
        }
        return info

    
    def reward(self): 

        reward = self.state_counts[S]*50
        vaccination_cost = (self.state_counts[V]*10)**2
        reward -= self.state_counts[I]*30
        reward-=vaccination_cost

        return reward
    
    def simple_reward(self): 
        reward = -self.state_counts[I]*30

        return reward
    
    
    def render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        state_info = {
            S: ("S", (255, 255, 255)),  
            I: ("I", (255, 0, 110)),           
            R: ("R", (131, 56, 236)),    
            V: ("V", (251, 86, 7))     
        }
        pix_square_size = self.window_size / self.size
        for x in range(self.size):
            for y in range(self.size):
                state = self.grid[x, y]
                pygame.draw.rect(
                    canvas,
                    state_info[state][1],
                    pygame.Rect(
                        pix_square_size * x, pix_square_size * y,
                        pix_square_size, pix_square_size
                    )
                )

        pygame.draw.circle(
            canvas,
            (58, 134, 255),
            (self.agent_position[0] * pix_square_size + pix_square_size / 2,
            self.agent_position[1] * pix_square_size + pix_square_size / 2),
            pix_square_size / 4
        )
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        
    
    
    
   