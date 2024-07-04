//
//  DefineTaskStruct.swift
//  Fruit Navigator
//
//  Created by roozbeh.kiani on 9/26/18.
//  Copyright © 2018 lab.kianilab. All rights reserved.
//

import UIKit
import Foundation

// keep task parameters as simple as possible. these parameters will be saved in plist
//files, which are suboptimal for versatile and deeply nested structures

struct TaskParam : Loopable {
    var DEBUG : Bool = false
    var name : String = "RaisinPuzzle"
    var subj_name : String = ""
    var algo_used : String = "Prim_modified"
    var entrance_mode : String = "top"
    var maze_height : Int = 7
    var maze_width : Int = 7
    var use_preset_ratio : Bool = false     //to be saved in the data, but will not appear on the menu
    var symmetry_control : String = "random"
   

    var rew_dur : Double = 0.3
    
    var ITI : Double = 1   //inter-trial interval in micro-seconds
   
    var frame_norm_h : CGFloat = 0.05
    var frame_width : CGFloat = 0
    var random_maze_scale : Bool = false
    var maze_scale : Double = 1
    var random_maze_position : Bool = false
    
    var max_dur : Double = 30
    
    var screenshot_save : Bool = true
    var screenshot_quality : CGFloat = 0.25  //between 0 and 1, choose small values to save disk
    
    var video_frame_save : Bool = true
    var video_grayscale : Bool = true
    var video_quality : CGFloat = 0.5       //between 0 and 1, choose small values to save disk and wifi bandwidth
    

    //default init(). really on the values set above and do not redefine
    init () {
    }
    
    //there is no dynamic referencing of structure fields in Swift, unlike Matlab.
    //so this init function is necessary. it is cumersome but our best path!
    init(dict: [String: Any]) {
        set(dict: dict)
    }
    
    
    
    mutating func set(dict: [String: Any]) {
        self.DEBUG = dict["DEBUG"] as? Bool ?? self.DEBUG
        self.name = dict["name"] as? String ?? self.name
        self.subj_name = dict["subj_name"] as? String ?? self.subj_name
        self.algo_used = dict["algo_used"] as? String ?? self.algo_used
        self.entrance_mode = dict["entrance_mode"] as? String ?? self.entrance_mode
        self.maze_width = dict["maze_width"] as? Int ?? self.maze_width
        self.maze_height = dict["maze_height"] as? Int ?? self.maze_height
        
        self.symmetry_control = dict["symmetry_control"] as? String ?? self.symmetry_control
       
        self.ITI = dict["ITI"] as? Double ?? self.ITI
     
        self.frame_norm_h = dict["frame_norm_h"] as? CGFloat ?? self.frame_norm_h
        self.frame_width = dict["frame_width"] as? CGFloat ?? self.frame_width
        self.random_maze_scale = dict["random_maze_scale"] as? Bool ?? self.random_maze_scale
        self.maze_scale = dict["maze_scale"] as? Double ?? self.maze_scale
        self.random_maze_position = dict["random_maze_position"] as? Bool ?? self.random_maze_position
        self.max_dur = dict["max_dur"] as? Double ?? self.max_dur
        
        self.screenshot_save = dict["screenshot_save"] as? Bool ?? self.screenshot_save
        self.screenshot_quality = dict["screenshot_quality"] as? CGFloat ?? self.screenshot_quality
        self.video_frame_save = dict["video_frame_save"] as? Bool ?? self.video_frame_save
        self.video_grayscale = dict["video_grayscale"] as? Bool ?? self.video_grayscale
        self.video_quality = dict["video_quality"] as? CGFloat ?? self.video_quality
    }
    
    


    
   
}



