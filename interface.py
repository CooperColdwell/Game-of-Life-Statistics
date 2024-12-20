import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
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

        # Initialize plot data
        self.steps_data = []
        self.white_cells_data = []
        
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
        ttk.Label(controls, text="Run Name:").grid(row=1, column=0, pady=5)
        self.filename_var = tk.StringVar(value="data")
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
        # Setup plot
        self.setup_plot(controls)      
  
        self.draw_grid()

    def setup_plot(self, parent):
        # Create figure and axis
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(bottom=0.15)  # Add bottom margin for x-label
        self.ax.set_xlabel('Step')
        self.ax.set_ylabel('Live Cells')
        self.ax.grid(True)
        
        # Create canvas
        self.plot_canvas = FigureCanvasTkAgg(self.figure, parent)
        self.plot_canvas.get_tk_widget().grid(row=5, column=0, columnspan=2, pady=10)
        
        # Initialize empty plot
        self.line, = self.ax.plot([], [])
        self.ax.set_xlim(0, 100)  # Initial x-axis range
        self.ax.set_ylim(0, 100)  # Initial y-axis range
        
    def update_plot(self):
        # Update data
        self.steps_data.append(self.current_step)
        self.white_cells_data.append(self.grid.get_white_cell_count())
        
        # Update plot
        self.line.set_data(self.steps_data, self.white_cells_data)
        
        # Adjust axis limits if needed
        if self.current_step >= self.ax.get_xlim()[1]:
            self.ax.set_xlim(0, self.current_step * 1.5)
        
        max_white_cells = max(self.white_cells_data) if self.white_cells_data else 100
        if max_white_cells >= self.ax.get_ylim()[1]:
            self.ax.set_ylim(0, max_white_cells * 1.2)
        
        self.figure.canvas.draw()
         
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
        self.update_plot()
    
    def run_simulation(self):
        if not self.running:
            self.running = True
            self.grid.save_initial_state(self.cell_size, self.filename_var.get())
            self.simulate_steps()

    def step_simulation(self, run_loop=False):

        self.grid.step()
        self.grid.record_state()
        if not run_loop:
            steps_remaining = int(self.timesteps_var.get())
            self.timesteps_var.set(str(steps_remaining - 1)) # decrement steps remaining
        if (self.current_step+1) % 10 == 0:
            self.grid.save_data(intermed=True)
        self.current_step += 1 # increment step counter
        self.current_step_var.set(str(self.current_step)) # increment current step
        self.draw_grid()
        self.update_plot()    
    def pause_simulation(self):
        self.running = False
    
    def simulate_steps(self):
        if self.running:
            try:
                steps_remaining = int(self.timesteps_var.get())
                if steps_remaining > 0:
                    self.step_simulation(run_loop=True)
                    self.timesteps_var.set(str(steps_remaining - 1)) # decrement steps remaining
                    self.root.after(100, self.simulate_steps) ## DELAY HERE: THIS IS HOW LONG BETWEEN STEPS
                else:
                    self.running = False
            except ValueError:
                self.running = False

    def save_data(self):
        self.grid.save_data(self.cell_size)

    def reset_simulation(self):
        self.grid = StateGrid(self.grid_size) # create a fresh StateGrid object
        self.running = False
        self.current_step = 0  # Reset step counter
        self.current_step_var.set("0")  # Update step counter display
        self.timesteps_var.set("100")

        # Reset plot data and clear plot
        self.steps_data = []
        self.white_cells_data = []
        self.line.set_data([], [])
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.figure.canvas.draw()
        self.draw_grid()

def main():
    root = tk.Tk()
    app = SimulationInterface(root)
    root.mainloop()

if __name__ == "__main__":
    main()