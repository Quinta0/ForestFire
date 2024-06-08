# Forest Fire Simulation
## Scope of the Project
The Forest Fire Simulation project aims to model the spread of forest fires over a grid representing a forest landscape. The simulation accounts for various factors such as different vegetation types, terrain elevation, wind speed and direction, humidity levels, and spontaneous ignition. This model helps to understand the dynamics of forest fires and can be used for educational purposes or to inform fire management strategies.

## Contents
1) Initialize Grid: Creates a grid with different cell types representing various forest components such as water bodies, dry grass, dense trees, burning areas, and normal forest.
2) Plot Grid: Visualizes the current state of the forest grid.
3) Neighbours: Identifies and returns the neighboring cells of a given point in the grid.
4) Propagate: Manages fire propagation based on cell type and the state of neighboring cells.
5) Update Grid: Updates the state of the grid based on fire propagation rules.
6) Simulate: Orchestrates the entire simulation over multiple iterations, updating and visualizing the grid.

## How to Run the Project
**Prerequisites**:
- Python 3.x
- Jupyter Notebook
- Necessary libraries: numpy, matplotlib

## Steps
1) Clone the Repository

````bash
git clone https://github.com/yourusername/forest_fire_simulation.git
cd forest_fire_simulation
````
2) Install Dependencies
````bash
pip install -r requirements.txt
````

3) Run the Jupyter Notebook
````bash
jupyter notebook forest_fire_simulation_complete.ipynb\
````

4) Execute the Notebook Cells
   
  Open the `forest_fire_simulation_complete.ipynb` notebook and run each cell sequentially to see the simulation in action.

## Project Structure
- forest_fire_simulation_complete.ipynb: The main Jupyter Notebook containing the implementation and simulation of the forest fire model.
- Forest Fire Simulation.pdf: Documentation detailing the algorithms and approaches used in the simulation.
- README.md: This file.

### Simulation Details

#### Grid Initialization

- **Function**: `initialize_grid(m, n, prob_tree, prob_burning, prob_water, prob_dry_grass, prob_dense_trees)`
- **Purpose**: Initializes the grid with different types of cells.
- **Process**: Assigns cell types based on specified probabilities.

#### Plotting the Grid

- **Function**: `plot_grid(grid)`
- **Purpose**: Visualizes the forest grid using different colors for different cell types.

#### Neighbor Identification

- **Function**: `neighbours(grid, i, j)`
- **Purpose**: Identifies the neighboring cells for a given cell in the grid.

#### Fire Propagation

- **Function**: `propagate(grid, i, j, p, pstart)`
- **Purpose**: Determines the next state of a cell based on its current state and the states of its neighbors.
- **Factors**:
    - Spontaneous Ignition Probability (`pstart`)
    - Fire Propagation Probability (`p`)
    - Wind Speed and Direction
    - Terrain Elevation
    - Humidity Levels

#### Updating the Grid

- **Function**: `update_grid(grid, p, pstart, w_speed, w_direction, terrain, humidity_grid)`
- **Purpose**: Updates the entire grid state by applying the fire propagation rules to each cell.

#### Running the Simulation

- **Function**: `simulate(m, n, iterations, p, pstart, w_speed, w_direction, terrain, humidity_grid)`
- **Purpose**: Runs the simulation for a specified number of iterations, updating and plotting the grid at each step.

### Conclusions

- **Markov Chains**: The simulation demonstrates that each cell in the grid can be considered a separate Markov Chain, with future states depending solely on the current state and the states of neighboring cells.
- **Influence of Factors**:
    - **Wind**: Wind speed and direction significantly affect the spread of fire.
    - **Vegetation**: Different types of vegetation have varying flammability, influencing fire spread probabilities.
    - **Terrain**: Terrain elevation impacts the speed and direction of fire spread, with fire spreading faster uphill.
    - **Humidity**: Higher humidity levels reduce fire spread probabilities, while lower humidity increases them.
    - **Randomness**: Random number generators introduce probabilistic behavior, making the simulation stochastic.

