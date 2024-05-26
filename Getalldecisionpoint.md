# Monkey's Trajectory Data Generation 

## Overview
This Python script processes maze trial data stored in MATLAB `.mat` files, determining decision points based on the trajectory of subjects through the maze, which is filter at more than two directions that monkey made, because we focus on analyzing monkey making  decison at branching point. 

## Functions

### `determine_maze_size(mat_file_name)`
- **Purpose**: Determines the size of the maze based on the date extracted from the filename.
- **Input**: Filename string containing a date.
- **Output**: Returns `6` for a 6x6 maze or `7` for a 7x7 maze depending on whether the date is before or after March 18, 2021.

### `extract_binary_strings(mazewall)`
- **Purpose**: Extracts binary strings representing the state of the maze walls from the given data structure.
- **Input**: Maze wall data structure from the `.mat` file. for e.g. mazewall = {
    'hidden': np.array([[[1, 0, 1, 1],
                         [0, 1, 1, 0],
                         [1, 0, 0, 1]]])
}

- **Output**: List of binary strings that indicate the state of walls in the maze. e.g.['1011', '0110', '1001']

- 


### `coordinate_to_binary_string_index(coord, maze_size)`
- **Purpose**: Calculates the linear index for a coordinate pair within the maze, facilitating the mapping of positions to binary wall states.
- **Input**: Tuple of coordinates (x, y) and the size of the maze.
- **Output**: Linear index as an integer. 

### `determine_direction(curr, next_pos)`
- **Purpose**: Determines the direction of movement between two consecutive coordinate points in the maze.
- **Input**: Current and next coordinate pairs. 
- **Output**: A string representing the direction (`'right'`, `'left'`, `'up'`, `'down'`, or `'stay'`). e.g. If curr = (1, 2) and next_pos = (2, 2), since delta_x = 1 (increase in x, no change in y), the output is "right".

### `process_single_file(mat_file_path, output_directory)`
- **Purpose**: Processes a single `.mat` file, extracts trial data, trajectory, wall states, and computes decision points where subjects have more than one path to choose from.
- **Input**: Path to the `.mat` file and an output directory path for saving results.
- **Output**: Text files are created in the specified directory with detailed decision points for each trial.
- Branching Point Filtering:

As part of processing each trial, the function identifies decision points by analyzing binary strings that represent the open or closed states of paths at each position in the maze. A position is marked as a decision point if there are multiple open paths (more than one 1 in the binary string), reflecting potential decision-making junctures for the subject navigating the maze. These decision points are then selectively recorded in the output files, providing a focused dataset for further behavioral analysis

