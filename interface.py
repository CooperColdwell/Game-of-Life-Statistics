import tkinter as tk
from tkinter import ttk
from playground import StateGrid

class SimulationInterface:
    def __init__(self, root, visible_size=50, grid_size=1000, cell_size=15):
        self.root = root
        self.root.title("Game of Life Interface")
        
        # Initialize simulation state
        self.grid_size = grid_size
        self.visible_size = visible_size
        self.cell_size = cell_size  # pixels per cell
        self.center_offset = (self.grid_size - self.visible_size) // 2 # the visible portion of the grid
        self.running = False
        self.current_step = 0

        # Create initial grid object
        self.grid = StateGrid(self.grid_size)

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

        # Current step counter
        ttk.Label(controls, text="Current step:").grid(row=3, column=0, pady=5)
        self.current_step_var = tk.StringVar(value="0")
        ttk.Label(controls, textvariable=self.current_step_var).grid(row=3, column=1, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(controls)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="RUN", command=self.run_simulation).grid(row=0, column=0, padx=2)
        ttk.Button(button_frame, text="STEP", command=self.step_simulation).grid(row=0, column=1, padx=2)
        ttk.Button(button_frame, text="PAUSE", command=self.pause_simulation).grid(row=0, column=2, padx=2)
        ttk.Button(button_frame, text="SAVE", command=self.save_data).grid(row=1, column=0, padx=2, pady=5)
        ttk.Button(button_frame, text="RESET", command=self.reset_simulation).grid(row=1, column=1, padx=2, pady=5)
        
        self.draw_grid()
        
    def draw_grid(self):
        self.canvas.delete("all")
        visible_region = self.grid.get_visible_region(self.center_offset, self.visible_size)
        
        for i in range(self.visible_size):
            for j in range(self.visible_size):
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                
                color = 'white' if visible_region[i][j] == 1 else 'black'
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='gray')
        
        # Update white cell counter
        self.white_cell_var.set(str(self.grid.get_white_cell_count()))
        
    def on_canvas_click(self, event):
        # Convert click coordinates to grid coordinates
        grid_x = event.x // self.cell_size + self.center_offset
        grid_y = event.y // self.cell_size + self.center_offset
        
        # Flip cell state
        self.grid.flip_cell(grid_x, grid_y)
        self.draw_grid()
    
    def run_simulation(self):
        if not self.running:
            self.running = True
            self.grid.save_initial_state(self.filename_var.get(), self.cell_size)
            self.simulate_steps()

    def step_simulation(self):
        self.grid.step()
        self.grid.record_state()
        self.current_step += 1 # increment step counter
        self.current_step_var.set(str(self.current_step)) # increment current step
        self.draw_grid()
        # self.running = False ## DEBUG/DEV: pauses after each step so I can check operability without getting stuck in loop
    
    def pause_simulation(self):
        self.running = False
    
    def simulate_steps(self):
        if self.running:
            try:
                steps_remaining = int(self.timesteps_var.get())
                if steps_remaining > 0:
                    self.step_simulation()
                    self.timesteps_var.set(str(steps_remaining - 1)) # decrement steps remaining
                    self.root.after(100, self.simulate_steps)
                else:
                    self.running = False
            except ValueError:
                self.running = False

    def save_data(self):
        self.grid.save_data(self.filename_var.get(), self.cell_size)

    def reset_simulation(self):
        self.grid = StateGrid(self.grid_size) # create a fresh StateGrid object
        self.running = False
        self.current_step = 0  # Reset step counter
        self.current_step_var.set("0")  # Update step counter display
        self.draw_grid()

def main():
    root = tk.Tk()
    app = SimulationInterface(root)
    root.mainloop()

if __name__ == "__main__":
    main()