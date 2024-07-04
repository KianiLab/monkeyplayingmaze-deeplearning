//
//  Maze.swift
//  May31mazegamescene
//
//  Created by Xinglan Zhao on 5/31/24.
//

import Foundation
import SpriteKit



enum CellEdge : Int {
    case right = 0
    case top = 1
    case left = 2
    case bottom = 3
    case none = 4
    
    static func index2Edge(index : Int) -> CellEdge {
        var e : CellEdge
        switch index {
        case right.rawValue:
            e = right
        case top.rawValue:
            e = top
        case left.rawValue:
            e = left
        case bottom.rawValue:
            e = bottom
        default:
            e = none
        }
        return e
    }
}


struct MazePosition : CustomStringConvertible {
    var x: Int
    var y: Int
    
    init() {
        x = 0
        y = 0
    }
    
    init(x: Int, y: Int) {
        self.x = x
        self.y = y
    }
    
    var description : String {
        return "[\(x), \(y)]"
    }
}

//struct MazeSize : CustomStringConvertible {
//    var width: Int
//    var height: Int
//    
//    init(width: Int, height: Int) {
//        self.width = width
//        self.height = height
//    }
//    
//    var description : String {
//        return "[\(width), \(height)]"
//    }
//}


class MazeCell : Loopable {
    var size : MazeSize = MazeSize(width: 6, height: 6)
 

    static let edge_list : [CellEdge] = [.right, .top, .left, .bottom]
   

    let walls = [SKSpriteNode(imageNamed: "whitewall_right"),
                 SKSpriteNode(imageNamed: "whitewall_top"),
                 SKSpriteNode(imageNamed: "whitewall_left"),
                 SKSpriteNode(imageNamed: "whitewall_bottom")]
    
    func addWallsToNode(_ node: SKNode) {
        for wall in walls {
            if wall.parent != nil {
                wall.removeFromParent() // Remove the wall from its current parent if it has one
            }
            node.addChild(wall) // Add the wall to the new node
        }
    }

    
}
class Maze : Loopable {
    var size: MazeSize
    var board: [[MazeCell]]
    var entrance_pos : MazePosition = MazePosition(x: 0, y: 0)
    var exit_pos : MazePosition = MazePosition(x: 0, y: 0)

    init(width: Int, height: Int) {
        self.size = MazeSize(width: width, height: height)
        self.board = []

        for _ in 0..<height {
            var row: [MazeCell] = []
            for _ in 0..<width {
                let cell = MazeCell()
                row.append(cell)
            }
            board.append(row)
        }
    }
}



//class Maze : Loopable {
//    var size : MazeSize = MazeSize(width: 0, height: 0)
//    var board = [[MazeCell]]()
//    
//}
