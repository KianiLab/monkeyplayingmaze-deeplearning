# image data generation 

## Functions Overview
Converts textual descriptions of mazes (including wall placements and path layouts) into non texctural image files. These images are then used as inputs for CNN models that learn to navigate or solve the mazes.

### `determine_maze_size(mat_file_name)`
**Purpose**: Calculates the size of the maze based on the date extracted from the file name.
**Input**: 
  - `mat_file_name`: Filename string that includes a date.
**Output**: 
  - Integer representing the size of the maze (either 6x6 or 7x7).

### `generate_ascii_maze(maze_size)`
**Purpose**: Initializes an ASCII grid representation of the maze.
**Input**: 
  - `maze_size`: Integer that determines the dimensions of the maze.
**Output**: 
  - 2D list containing the ASCII characters forming the maze structure.

### `add_walls_to_maze(maze, binary_strings, maze_size)`
**Purpose**: Populates the maze with walls based on binary encoding.
**Input**: 
  - `maze`: 2D list representing the ASCII maze.
  - `binary_strings`: List of strings representing wall states as binary codes.
  - `maze_size`: Integer indicating the size of the maze.
**Output**: 
  - None (modifies the maze in-place).

### `extract_and_save_ascii_maze(mat_file_name, trial_index, output_directory, binary_strings, entrance, exit)`
**Purpose**: Coordinates the processes to generate and save ASCII representations of mazes.
**Input**: 
  - `mat_file_name`: The base name of the `.mat` file.
  - `trial_index`: Index of the trial in the dataset.
  - `output_directory`: Path to the directory where outputs will be saved.
  - `binary_strings`: Binary string data for wall positions.
  - `entrance`: Tuple representing the coordinates of the maze entrance.
  - `exit`: Tuple representing the coordinates of the maze exit.
**Output**: 
  - None (saves files to disk).

### `maze_to_image(maze, entrance, exit, wall_color, path_color, entrance_color, exit_color, wall_thickness, path_to_wall_ratio, horizontal_stretch_factor)`
**Purpose**: Converts an ASCII maze into a graphical image format.
**Input**: 
  - `maze`: 2D list of the ASCII maze.
  - `entrance`: Tuple for the entrance coordinates.
  - `exit`: Tuple for the exit coordinates.
  - `wall_color`: Color of the walls.
  - `path_color`: Background color of the maze paths.
  - `entrance_color`: Color for the entrance.
  - `exit_color`: Color for the exit.
  - `wall_thickness`: Thickness of the walls in pixels.
  - `path_to_wall_ratio`: Ratio of the path width to the wall thickness.
  - `horizontal_stretch_factor`: Factor by which the horizontal dimension is stretched.
**Output**: 
  - `PIL.Image`: Image object of the maze.

### `convert_coordinate(mat_entrance, mat_exit, mat_file_name)`
**Purpose**: Converts MATLAB coordinates for the maze entrance and exit into coordinates usable in the ASCII and image representations.
**Input**: 
  - `mat_entrance`: Tuple of MATLAB coordinates for the entrance.
  - `mat_exit`: Tuple of MATLAB coordinates for the exit.
  - `mat_file_name`: Filename used to determine maze size.
**Output**: 
  - Two tuples representing the converted coordinates for entrance and exit.

### `process_single_file(mat_file_path, output_directory)`
**Purpose**: Processes each `.mat` file to extract data and generate output for each trial.
**Input**: 
  - `mat_file_path`: Path to the `.mat` file.
  - `output_directory`: Directory where output files will be stored.
**Output**: 
  - None (generates ASCII and image files).

