import numpy as np
from PIL import Image
import os
from datetime import datetime

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

    def step(self):
        live_coords = np.where(self.state==1)
        live_coords_print = [(int(y), int(x)) for y,x in zip(live_coords[0], live_coords[1])] # numpy arrays start in the top left corner and are indexed by arr[row][col]
        print(live_coords_print)

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