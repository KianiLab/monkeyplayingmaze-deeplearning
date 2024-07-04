//
//  GameScene.swift
//  May31mazegamescene
//
//
//


import SpriteKit
import GameplayKit
import Foundation

struct MazeSize {
    var width: Int
    var height: Int
}



class GameScene: SKScene {
    //    private let maze = Maze()
    
    
    var cellWidth: CGFloat = 0
    var cellHeight: CGFloat = 0
    let jsonFilename:String
    
    
    
    var frame_norm_h : CGFloat = 0.05
    private let screen_size = UIScreen.main.bounds
    private var scale=param.random_maze_scale ? Double.random(in: 0.3...1) : param.maze_scale
    private var maze: Maze
    var maze_size: MazeSize
    //    var maze_size = MazeSize(width: 6, height: 6)  // Define the maze size as 6x6
    //    var maze_size = MazeSize(width: param.maze_width, height:param.maze_height)//MazeSize(width: 10, height: 6)
    
    let maze_wall_norm_w: CGFloat = 0.1 // Wall thickness percentage of the cell width
    let maze_wall_norm_h: CGFloat = 0.1 // Wall height percentage of the cell height
    var walls: [[SKSpriteNode]] = []  // Array to store wall nodes
    private let obj_wallframe = [TaskGraphObj(tag: 50),
                                 TaskGraphObj(tag: 51)]
    private let blankscene = SKSpriteNode(color: .black, size: CGSize(width: 500, height: 500))
    //    private let wallframe = [SKSpriteNode(imageNamed: "wall-horiz1.png"),
    //                             SKSpriteNode(imageNamed: "wall-horiz1.png")]
    private var obj_walls = [TaskGraphObj]()
    // display scale of the maze, updated in every trial if random_maze_scale is true
    
    
    init(size: CGSize, jsonFilename:String,mazeWidth: Int=7, mazeHeight: Int=7) {
        self.jsonFilename = jsonFilename
        self.maze_size = MazeSize(width: mazeWidth, height: mazeHeight)  // Set maze size from parameters
        self.maze = Maze(width: mazeWidth, height: mazeHeight)  // Initialize the maze with given size
//        super.init(size: size)
        
        obj_walls = Array(repeating: TaskGraphObj(tag:100), count: mazeWidth * mazeHeight * 4)
        var tag: Int32 = 100
        for j in 0..<mazeHeight {
            for i in 0..<mazeWidth {
                for k in 0..<4 {
                    let index = (j * mazeWidth + i) * 4 + k
                    obj_walls[index] = TaskGraphObj(tag: tag)
                    tag += 1
                }
            }
        }
        
        super.init(size: size)
               
               loadWallStates()
    }
    
    
    //    // Required initializer for decoding (used when scenes are loaded from .sks files)
    //      required init?(coder aDecoder: NSCoder) {
    //          // Provide default values for decoding or handle appropriately
    //          let width = aDecoder.decodeInteger(forKey: "mazeWidth")
    //          let height = aDecoder.decodeInteger(forKey: "mazeHeight")
    //          self.maze_size = MazeSize(width: width > 0 ? width : 6, height: height > 0 ? height : 6)
    //          self.maze = Maze(width: maze_size.width, height: maze_size.height)
    //          super.init(coder: aDecoder)
    //      }
    
    required init?(coder aDecoder: NSCoder) {
        // Decode the maze width and height, providing default values if not present.
        let width = aDecoder.decodeInteger(forKey: "mazeWidth")
        let height = aDecoder.decodeInteger(forKey: "mazeHeight")
        self.maze_size = MazeSize(width: width > 0 ? width : 6, height: height > 0 ? height : 6)
        self.maze = Maze(width: maze_size.width, height: maze_size.height)
        
        // Decode the jsonFilename or provide a default value if not present.
        if let filename = aDecoder.decodeObject(forKey: "jsonFilename") as? String {
            self.jsonFilename = filename
        } else {
            // Provide a default filename or handle error
            self.jsonFilename = "error_filename.json" // Modify as needed
        }
        
        super.init(coder: aDecoder)
    }
    
    
    
    struct PhysicsCategory {
        static let none : UInt32 = 0
        static let fruit : UInt32 = 0b001
        static let goal : UInt32 = 0b010
        static let wall : UInt32 = 0b100
    }
    
    
    private func loadWallStates() {
//        guard let url = Bundle.main.url(forResource: jsonFilename, ofType: "json") else {
//            print("Error: Could not find JSON file in bundle.")
//            fatalError("Failed to load wall states JSON because the file was not found in the bundle.")
//        }
        // Split the filename and extension
                let components = jsonFilename.split(separator: ".")
                guard components.count == 2 else {
                    print("Error: Invalid JSON filename format.")
                    return
                }
                
                let resourceName = String(components[0])
                let resourceExtension = String(components[1])
        
                print("Resource Name: \(resourceName)")
                print("Resource Extension: \(resourceExtension)")
                
                // Attempt to load the JSON file
                guard let url = Bundle.main.url(forResource: resourceName, withExtension: resourceExtension) else {
                    print("Error: Could not find JSON file in bundle.")
                    fatalError("Failed to load wall states JSON because the file was not found in the bundle.")
                }
        
        
        
        
        do {
            let data = try Data(contentsOf: url)
            print("JSON data loaded successfully.")
            
            guard let jsonObject = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any],
                  let wallStates = jsonObject["wall_states"] as? [[String]],
                  let entranceArray = jsonObject["entrance"] as? [[Int]],
                  let exitArray = jsonObject["exit"] as? [[Int]],
                  !entranceArray.isEmpty,
                  !exitArray.isEmpty,
                  entranceArray[0].count == 2,
                  exitArray[0].count == 2 else {
                print("Error: JSON format is incorrect.")
                return
            }
            
          
            let entrance = (entranceArray[0][0], entranceArray[0][1])
            let exit = (exitArray[0][0], exitArray[0][1])
            print("Wall States:", wallStates)
            print("Entrance:", entrance)
            print("Exit:", exit)
            // Calculate cell dimensions
               cellWidth = self.size.width * (1 - maze_wall_norm_w) / CGFloat(maze_size.width)
               cellHeight = self.size.height * (1 - maze_wall_norm_h) * (1 - frame_norm_h * 2) / CGFloat(maze_size.height)
               
            self.maze = Maze(width: wallStates[0].count, height: wallStates.count) // Initialize the maze with sizes based on wallStates dimensions
            
            setupScene(with: wallStates, entrance: entrance, exit: exit)
            
        
        } catch {
            print("Error: \(error.localizedDescription)")
            fatalError("Failed to load or parse wall states JSON.")
        }
    }
   
    
    //    //load
    //    private func loadWallStates() {
    //        //"Hermann_20210217T155110_detour_nodes"
    //        guard let url = Bundle.main.url(forResource: jsonFilename, withExtension: "json") else {
    //            print("Error: Could not find JSON file in bundle.")
    //            fatalError("Failed to load wall states JSON because the file was not found in the bundle.")
    //        }
    //
    //        do {
    //            let data = try Data(contentsOf: url)
    //            print("JSON data loaded successfully.")
    ////            guard let jsonObject = try JSONSerialization.jsonObject(with: data, options: []) as?
    ////                    let wallStates = jsonObject["wall_states"] as? [[String]]
    ////            }
    //            guard let jsonObject = try JSONSerialization.jsonObject(with: data, options: []) as? [[String]] else {
    //                            print("Error: Could not cast JSON object to expected type.")
    //                            fatalError("Failed to parse wall states JSON: Invalid format.")
    //                        }
    //            print("Parsed JSON object: \(jsonObject)")
    //            self.maze = Maze(width: jsonObject[0].count, height: jsonObject.count) // Initialize the maze with sizes
    //            setupScene(with: jsonObject)
    //        } catch {
    //            print("Error: \(error.localizedDescription)")
    //            fatalError("Failed to load or parse wall states JSON.")
    //        }
    //    }
    
    override func didMove(to view: SKView) {
        super.didMove(to: view)
//        loadWallStates()
         
        //        setupScene(add_nodes: true)
        adjustSceneScaleAndPosition()
        print("Scene loaded successfully")
        
    }
    

    
    private func adjustSceneScaleAndPosition() {
        let screen_width = Double(UIScreen.main.bounds.width)
        let screen_height = Double(UIScreen.main.bounds.height)
        
        // Assuming 'scale' is already defined and adjusted based on your game's needs
        self.size = CGSize(width: screen_width / scale, height: screen_height / scale)
        
        let anchor_xpos: Double
        let anchor_ypos: Double
        
        //        if param.random_maze_position {
        anchor_xpos = Double.random(in: 0...1-scale)
        anchor_ypos = Double.random(in: 0...1-scale)
        //        } else {
        //            anchor_xpos = (1-scale)/2
        //            anchor_ypos = (1-scale)/2
        ////
        
        self.anchorPoint = CGPoint(x: anchor_xpos, y: anchor_ypos)
    }
    
    
    private func setupScene(with wallStates: [[String]], entrance: (Int, Int), exit: (Int, Int)) {
        // Compute the size of a single cell, excluding the wall boundary
        let cell_width = self.size.width * (1 - maze_wall_norm_w) / CGFloat(maze_size.width)
        let cell_height = self.size.height * (1 - maze_wall_norm_h) * (1 - param.frame_norm_h * 2) / CGFloat(maze_size.height)
        
        // Update the maze size
        // Print the dimensions of the maze and each cell
//        print("Maze width: \(self.size.width) pixels")
//        print("Maze height: \(self.size.height) pixels")
//        print("Cell width: \(cell_width) pixels")
//        print("Cell height: \(cell_height) pixels")
        if param.symmetry_control == "preset ratio, size = 6x6" {
            maze_size = MazeSize(width: 6, height: 6)
        } else if param.symmetry_control == "preset ratio, size = 7x7" {
            maze_size = MazeSize(width: 7, height: 7)
        } else {
            maze_size = MazeSize(width: param.maze_width, height: param.maze_height)
        }
        for i in 0..<maze.size.width {
            for j in 0..<maze.size.height {
                for wall in maze.board[j][i].walls {
                    if wall.parent != nil {
                        wall.removeFromParent()
                    }
                }
            }
        }
        
//        
//        print("Setting up scene with cell dimensions: \(cell_width)x\(cell_height)")
//        print("Setting up scene with maze dimensions: \(maze_size.width) x \(maze_size.height)")
//        
        for j in 0..<maze_size.height {
            for i in 0..<maze_size.width {
                if (i,j)==entrance{
                    if j == 5 {
                        let entranceNode = SKSpriteNode(color: .red, size: CGSize(width: cell_width-19, height: cell_height-19))
                        let xPosition: CGFloat
                        if i==6{
                            
                            xPosition = self.size.width - (cell_width / 2)
                        }else{
                            xPosition = self.size.width - (1.1 * (cell_width / 2) + 2.2 * CGFloat(6 - i) * (cell_width / 2))
                            
                        }
                        
                        
                        let yPosition = self.size.height - cell_height
                        
                        entranceNode.position = CGPoint(x: xPosition, y: yPosition)
                        entranceNode.zPosition = 1.5  // Make sure it is visible above the background but below other potential overlay elements.
                        
                        addChild(entranceNode)
                        
                    }
                    if j == 6 {
                        let entranceNode = SKSpriteNode(color: .red, size: CGSize(width: cell_width-19, height: cell_height-19))
                        let xPosition: CGFloat
                        if i==6{
                            
                            xPosition = self.size.width - (cell_width / 2)
                        }else{
                            xPosition = self.size.width - (1.1 * (cell_width / 2) + 2.2 * CGFloat(6 - i) * (cell_width / 2))
                            
                        }
                        
                        
                        let yPosition = self.size.height - cell_height
                        
                        entranceNode.position = CGPoint(x: xPosition, y: yPosition)
                        entranceNode.zPosition = 1.5  // Make sure it is visible above the background but below other potential overlay elements.
                        
                        addChild(entranceNode)
                        
                    }
                }
                
                if (i,j)==exit{
                    if j == 0 {
                        let exitNode = SKSpriteNode(color: .green, size: CGSize(width: cell_width-18, height: cell_height-18))
                       
                       
                        let xPosition: CGFloat
                        if i==6{
                            
                            xPosition = self.size.width - (cell_width / 2)
                        }else {
                            
                            xPosition = self.size.width - (1.1 * (cell_width / 2) + 2.2 * CGFloat(6 - i) * (cell_width / 2))
                            
                        }
                        let yPosition = cell_height
                        print("yPosition!: \(yPosition)")
                        exitNode.position = CGPoint(x: xPosition, y: yPosition)
                        
                        exitNode.zPosition = 0.5  // Set zPosition to be visible
                        addChild(exitNode)
                    }
                    
                }
                
                let mazestate_row = i
                let mazestate_col = j
                //                print("Mazeboard (\(j), \(i)) maps to Mazestate (\(mazestate_row), \(mazestate_col))")
                //
//                print("Wall States Array Dimensions: \(wallStates.count) rows, \(wallStates.first?.count ?? 0) columns")
//                print("Trying to access: Row \(mazestate_row), Column \(mazestate_col)")
//                
                
                // 根据 wallStates 中的信息，为每个单元格添加墙体
                //
                let state=wallStates[mazestate_row][mazestate_col]
                //                let p=wallStates[5][0]
                //            print("p:\(p)")
                
                //                print("Current wall state: \(state) for cell [\(j), \(i)]")
                //
                for (k, wall) in maze.board[j][i].walls.enumerated() {
                    let wallPresence = state[state.index(state.startIndex, offsetBy: k)]
                    //                    print("Index: \(k), Wall: \(wall), Presence: \(wallPresence)")
                    //
                    //
                    //
                    //                    print("Wall index \(k) (Direction: \(MazeCell.edge_list[k]))")
                    //
                    if wallPresence == "0" { // 0表示有墙
                        // 根据墙体位置调整尺寸和位置
                        var w,h, x, y: CGFloat
                        if MazeCell.edge_list[k] == .left || MazeCell.edge_list[k] == .right {
                            w = self.size.width * maze_wall_norm_w / CGFloat(maze.size.width) / 2
                            h = self.size.height * (1-param.frame_norm_h*2) / CGFloat(maze.size.height)
                            if j>0 && j<maze.size.height-1 {
                                h *= (1+maze_wall_norm_h)
                            }
                            y = self.size.height * (param.frame_norm_h + (1-param.frame_norm_h*2)*(CGFloat(j)+0.5)/CGFloat(maze.size.height))
                            if j==0 {
                                y += self.size.height * (maze_wall_norm_h/2)/CGFloat(maze.size.height) - 1
                            } else if j==maze.size.height-1 {
                                y -= self.size.height * (maze_wall_norm_h/2)/CGFloat(maze.size.height) - 1
                            }
                            if MazeCell.edge_list[k] == .left {
                                x = self.size.width * (CGFloat(i)+maze_wall_norm_w/4)/CGFloat(maze.size.width)
                            } else {
                                x = self.size.width * (CGFloat(i)+(1-maze_wall_norm_w/4))/CGFloat(maze.size.width)
                            }
                            
                            // top wall and bottom wall
                        } else {
                            w = self.size.width  / CGFloat(maze.size.width)
                            h = self.size.height * (1-param.frame_norm_h*2) * maze_wall_norm_h / CGFloat(maze.size.height) / 2
                            x = self.size.width * (CGFloat(i)+0.5)/CGFloat(maze.size.width)
                            if MazeCell.edge_list[k] == .bottom {
                                y = self.size.height * (param.frame_norm_h + (1-param.frame_norm_h*2)*(CGFloat(j)+maze_wall_norm_h/4)/CGFloat(maze.size.height))
                            } else {
                                y = self.size.height * (param.frame_norm_h + (1-param.frame_norm_h*2)*(CGFloat(j)+(1-maze_wall_norm_h/4))/CGFloat(maze.size.height))
                            }
                        }
                        
                        wall.size = CGSize(width: w, height: h)
                        wall.position = CGPoint(x: x, y: y)
                        wall.zPosition = 1
                        
                        //
                        //                        print("Adding wall at position (\(x), \(y)) to cell[\(j)][\(i)] with size (\(w), \(h))")
                        //
                        
                        addChild(wall)
                        let index = (j*maze.size.width+i)*4+k
                        obj_walls[index].setShape(shape: .RECTANGLE)
                        //                        print("Number of elements in obj_walls: \(obj_walls.count)")
                        
                        obj_walls[index].setSize(size: wall.size, scale: scale)
                        obj_walls[index].setColor(r: 200, g: 200, b: 200, a: 255)
                        obj_walls[index].setFill(fill: true)
                        
                        
                        
                        
                    } else {
                        wall.isHidden = true
                    }
                    
                    
                }
                
            }
        }
        
        
        
        
        
       
        
        
        
    }
    
    
}


        

