import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image
import os
from datetime import datetime

class SimulationInterface:
    def __init__(self, root, visible_size=50, grid_size=1000, cell_size=15):
        self.root = root
        self.root.title("Discrete-Time Simulation")
        
        # Initialize simulation state
        self.grid_size = grid_size
        self.visible_size = visible_size
        self.cell_size = cell_size  # pixels per cell
        self.state = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.center_offset = (self.grid_size - self.visible_size) // 2
        self.running = False
        self.white_cell_history = []
        
        self.setup_interface()
        
    def setup_interface(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Canvas for grid
        self.canvas = tk.Canvas(
            main_frame, 
            width=self.visible_size * self.cell_size,
            height=self.visible_size * self.cell_size,
            bg='white'
        )
        self.canvas.grid(row=0, column=0, rowspan=6, padx=5, pady=5)
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        
        # Control panel
        controls = ttk.Frame(main_frame, padding="5")
        controls.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Timesteps input
        ttk.Label(controls, text="Number of Timesteps:").grid(row=0, column=0, pady=5)
        self.timesteps_var = tk.StringVar(value="100")
        ttk.Entry(controls, textvariable=self.timesteps_var).grid(row=0, column=1, pady=5)
        
        # Filename input
        ttk.Label(controls, text="Save Filename:").grid(row=1, column=0, pady=5)
        self.filename_var = tk.StringVar(value="simulation_data")
        ttk.Entry(controls, textvariable=self.filename_var).grid(row=1, column=1, pady=5)
        
        # White cell counter
        ttk.Label(controls, text="White Cells:").grid(row=2, column=0, pady=5)
        self.white_cell_var = tk.StringVar(value="0")
        ttk.Label(controls, textvariable=self.white_cell_var).grid(row=2, column=1, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(controls)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="RUN", command=self.run_simulation).grid(row=0, column=0, padx=2)
        ttk.Button(button_frame, text="STEP", command=self.step_simulation).grid(row=0, column=1, padx=2)
        ttk.Button(button_frame, text="PAUSE", command=self.pause_simulation).grid(row=0, column=2, padx=2)
        ttk.Button(button_frame, text="SAVE", command=self.save_data).grid(row=1, column=0, padx=2, pady=5)
        ttk.Button(button_frame, text="RESET", command=self.reset_simulation).grid(row=1, column=1, padx=2, pady=5)
        
        self.draw_grid()
        
    def draw_grid(self):
        self.canvas.delete("all")
        for i in range(self.visible_size):
            for j in range(self.visible_size):
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                
                cell_state = self.state[i + self.center_offset][j + self.center_offset]
                color = 'white' if cell_state == 1 else 'black'
                
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='gray')
        
        # Update white cell counter
        white_cells = np.sum(self.state == 1)
        self.white_cell_var.set(str(white_cells))
        
    def on_canvas_click(self, event):
        # Convert click coordinates to grid coordinates
        grid_x = event.x // self.cell_size + self.center_offset
        grid_y = event.y // self.cell_size + self.center_offset
        
        # Flip cell state
        self.state[grid_y][grid_x] = 1 - self.state[grid_y][grid_x]
        self.draw_grid()
    
    def run_simulation(self):
        if not self.running:
            self.running = True
            self.save_initial_state()
            self.simulate_steps()
    
    def step_simulation(self):
        live_coords = np.where(self.state==1)
        live_coords_print = [(int(y), int(x)) for y,x in zip(live_coords[0], live_coords[1])] # numpy arrays start in the top left corner and are indexed by arr[row][col]
        print(live_coords_print)
        self.running = False
    
    def pause_simulation(self):
        self.running = False
    
    def simulate_steps(self):
        if self.running:
            try:
                steps_remaining = int(self.timesteps_var.get())
                if steps_remaining > 0:
                    self.step_simulation()
                    self.timesteps_var.set(str(steps_remaining - 1))
                    self.root.after(100, self.simulate_steps)
                else:
                    self.running = False
            except ValueError:
                self.running = False
    
    def save_initial_state(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_grid_image(f"{self.filename_var.get()}_initial_{timestamp}.png")
        self.white_cell_history = [np.sum(self.state == 1)]
    
    def save_data(self):
        if not os.path.exists('simulation_data'):
            os.makedirs('simulation_data')
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = self.filename_var.get()
        
        # Save current grid state
        self.save_grid_image(f"{base_filename}_final_{timestamp}.png")
        
        # Save white cell history
        with open(f"simulation_data/{base_filename}_history_{timestamp}.txt", 'w') as f:
            for i, count in enumerate(self.white_cell_history):
                f.write(f"Timestep {i}: {count}\n")
    
    def save_grid_image(self, filename):
        img = Image.new('RGB', (self.grid_size * self.cell_size, self.grid_size * self.cell_size), 'black')
        pixels = img.load()
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                color = (255, 255, 255) if self.state[i][j] == 1 else (0, 0, 0)
                for x in range(self.cell_size):
                    for y in range(self.cell_size):
                        pixels[j * self.cell_size + x, i * self.cell_size + y] = color
        
        img.save(f"simulation_data/{filename}")
    
    def reset_simulation(self):
        self.state = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.white_cell_history = []
        self.running = False
        self.draw_grid()

def main():
    root = tk.Tk()
    app = SimulationInterface(root)
    root.mainloop()

if __name__ == "__main__":
    main()