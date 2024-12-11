import numpy as np
from PIL import Image
import os
from datetime import datetime
from typing import Set, Tuple, Callable

class StateGrid:
    def __init__(self, grid_size=1000):
        self.grid_size = grid_size
        self.state = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.white_cell_history = []

    def get_cell(self, x, y):
        return self.state[y][x]

    def flip_cell(self, x, y):
        """ Flip the state of cell at state[y][x]"""
        self.state[y][x] = 1 - self.state[y][x]

    def get_white_cell_count(self):
        return np.sum(self.state == 1)

    def get_visible_region(self, center_offset, visible_size):
        """ Returns the specified region of the grid. For interface use. """
        start_x = center_offset
        start_y = center_offset
        end_x = start_x + visible_size
        end_y = start_y + visible_size
        return self.state[start_y:end_y, start_x:end_x]

    def _compute_neighborhood_sum(self, center_rows: np.ndarray, center_cols: np.ndarray) -> np.ndarray:
        """
        Compute the sum of each center position and its eight neighbors.
        
        This function uses clever broadcasting to compute sums for multiple centers simultaneously.
        Rather than looping through each center and its neighbors, we create offset arrays for all
        neighbor positions at once and use NumPy's advanced indexing to gather the values.
        """
        # Define relative positions for center and all 8 neighbors
        rel_positions = np.array([
            (0, 0),    # Center
            (-1, -1), (-1, 0), (-1, 1),  # Top row
            (0, -1),            (0, 1),   # Middle row
            (1, -1),  (1, 0),  (1, 1)     # Bottom row
        ])

        # Broadcasting to get all neighbor positions for all centers at once
        # Shape will be (num_centers, num_neighbors)
        neighbor_rows = center_rows[:, np.newaxis] + rel_positions[:, 0]
        neighbor_cols = center_cols[:, np.newaxis] + rel_positions[:, 1]

        # Create boundary mask - make sure locations fall within the actual grid
        valid_rows = (neighbor_rows >= 0) & (neighbor_rows < self.state.shape[0])
        valid_cols = (neighbor_cols >= 0) & (neighbor_cols < self.state.shape[1])
        valid_mask = valid_rows & valid_cols
        
        # Initialize sums array
        neighborhood_sums = np.zeros(len(center_rows))
        
        # Get values and compute sums, handling boundary conditions
        for i in range(len(rel_positions)):
            valid_positions = valid_mask[:, i]
            rows = neighbor_rows[valid_positions, i]
            cols = neighbor_cols[valid_positions, i]
            neighborhood_sums[valid_positions] += self.state[rows, cols]
        
        return neighborhood_sums

    def _cascade_neighbor_analysis(
        self,
        # condition_a: Callable[[np.ndarray], np.ndarray],
        # condition_b: Callable[[np.ndarray], np.ndarray],
        # starting_value: float = 1
    ) -> Set[Tuple[int, int]]:
        """
        Perform cascading neighborhood analysis:
            1. Find initial positions where array equals starting_value
            2. For each position, compute sum of position + neighbors
            3. Find neighbors that satisfy condition B
            4. For those neighbors, compute their neighborhood sums
            5. Return all unique positions that satisfy the conditions
            
        I originally planned to use np.where() to get relevant locations, then iterate for each location,
        but the response I got when I asked Anthropic's Claude made me realize I could handle all at once.

        Parameters:
        array: 2D numpy array
        condition_a: function that takes neighborhood sums and returns boolean mask
        condition_b: function that takes neighborhood sums and returns boolean mask
        starting_value: value to start the search from (default 1)
        
        Returns:
        set of (row, col) tuples for all positions satisfying the conditions
        """
        # Get positions of live cells
        live_rows, live_cols = np.where(self.state == 1)
        
        # Compute sums of initially live cells and their neighbors
        live_sums = self._compute_neighborhood_sum(live_rows, live_cols)
        
        # Get live cells that stay live in the next step
        still_alive = (live_sums==3) | (live_sums==4) # if local sum=3, live, if local sum=4, keep value

        # Create meshgrid of relative neighbor positions
        rel_positions = np.array([
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ])
        
        # Broadcasting magic: Add relative positions to each target position
        # This creates arrays of shape (num_targets, num_neighbors)
        neighbor_rows = live_rows[:, np.newaxis] + rel_positions[:, 0]
        neighbor_cols = live_cols[:, np.newaxis] + rel_positions[:, 1]
        
        # Create boundary mask
        valid_rows = (neighbor_rows >= 0) & (neighbor_rows < self.state.shape[0])
        valid_cols = (neighbor_cols >= 0) & (neighbor_cols < self.state.shape[1])
        valid_mask = valid_rows & valid_cols
        
        # Flatten and get unique neighbor positions -- i.e. don't repeat processing on shared neighbors
        valid_neighbor_rows = neighbor_rows[valid_mask]
        valid_neighbor_cols = neighbor_cols[valid_mask]
        unique_neighbors = np.unique(np.column_stack((valid_neighbor_rows, valid_neighbor_cols)), axis=0)
        # Compute local sums for dead neighbors of currently live cells
        dead_unique_neighbors = self.state[unique_neighbors[:,0], unique_neighbors[:,1]]==0
        neighbor_sums = self._compute_neighborhood_sum(unique_neighbors[dead_unique_neighbors, 0], unique_neighbors[dead_unique_neighbors, 1])

        # Get dead neighbors that come to life
        births = neighbor_sums == 3 # if local sum is 3, then that cell comes to life
        final_rows = unique_neighbors[dead_unique_neighbors, 0][births]
        final_cols = unique_neighbors[dead_unique_neighbors, 1][births]

        # Return the cells that should be alive in the next time step
        return np.concatenate((final_rows, live_rows[still_alive]),axis=0), np.concatenate((final_cols, live_cols[still_alive]), axis=0)

    def step(self):
        """ 
        'Egocentric approach' to checking GoL conditions (from the GoL Wikipedia page):
            1. if sum of all 9 fields in a neighborhood is 3, inner field state in next gen is life
                - 2 cases: 3 live cells around dead cell (birth), or 2 live cells around live cell (survival)
            2. if sum of all 9 fields is 4, the center cell keeps current state
                - live cell with 3 live neighbors: survive
                - dead cell with 4 live neighbors: stays dead
            3. any other sum: center field dies or stays dead
        This approach checks the neighborhood for ALL locations (like a sliding convolution), which I consider to be inefficient.
        The state of a dead cell surrounded by dead cells will not change, so we shouldn't bother checking them. Instead, check
        the state conditions for each live cell's neighborhood, and store the locations of each of its dead neighbors. After all
        live cells have been checked, check the stored dead neighbor locations. 
                (Using this method probably rules out using GPU acceleration, but I don't think my grid is big enough to matter)

        The above rules don't work for cells on the borders of the "playground" unless periodic boundary conditions/toroidal array
        is used. This implementation uses rigid boundaries (everything outside the grid is dead), so edge cells must be checked 
        normally.
        """
        # Get cells that should be alive in the next time step
        live_rows, live_cols = self._cascade_neighbor_analysis()

        # Wipe the state grid
        self.state = np.zeros_like(self.state)

        # Set the relevant cells to live
        self.state[live_rows,live_cols] = 1

    def record_state(self):
        self.white_cell_history.append(self.get_white_cell_count())

    def save_data(self, filename, cell_size=5):
        if not os.path.exists('simulation_data'):
            os.makedirs('simulation_data')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save grid state as image
        self._save_grid_image(f"{filename}_final_{timestamp}.png", cell_size)

        # Save white cell history
        with open(f"simulation_data/{filename}_history_{timestamp}.txt", 'w') as f:
            for i, count in enumerate(self.white_cell_history):
                f.write(f"Timestep {i}: {count}\n")

    def save_initial_state(self, filename, cell_size=5):
        if not os.path.exists('simulation_data'):
            os.makedirs('simulation_data')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._save_grid_image(f"{filename}_initial_{timestamp}.png", cell_size)
        self.white_cell_history = [self.get_white_cell_count()]

    def _save_grid_image(self, filename, cell_size):
        img = Image.new('RGB', (self.grid_size * cell_size, self.grid_size * cell_size), 'black')
        pixels = img.load()

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                color = (255, 255, 255) if self.state[i][j] == 1 else (0, 0, 0)
                for x in range(cell_size):
                    for y in range(cell_size):
                        pixels[j * cell_size + x, i * cell_size + y] = color

        img.save(f"simulation_data/{filename}")