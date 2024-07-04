
//
//  TaskGraphObject.swift
//  Fruit Navigator
//
//  Created by roozbeh.kiani on 12/8/18.
//  Copyright © 2018 lab.kianilab. All rights reserved.
//

import Foundation
import UIKit
import SpriteKit

enum ObjectShape : Int32 {
    case ELLIPSE = 0
    case RECTANGLE = 1
    case ARROW = 2
    case IMAGE = 3
}

class TaskGraphObj : Loopable {
    struct Position {
        var x, y, w, h : Double
    }
    struct Color {
        var r, g, b : UInt8
    }
    
    struct Properties {
        var tag : Int32 = -1
        var shape : ObjectShape = .ELLIPSE
        var x : Double = 0
        var y : Double = 0
        var w : Double = 0
        var h : Double = 0
        var r : Int32 = 0
        var g : Int32 = 0
        var b : Int32 = 0
        var a : Int32 = 0
        var fill : Int16 = 0
        var visible : Int16 = 0
    }
    
    var properties : Properties = Properties()
    
    init(tag: Int32) {
        properties.tag = tag
    }
    
    init(tag: Int32, shape: ObjectShape, position: CGPoint, size: CGSize, color: UIColor, fill: Bool, visible: Bool) {
        properties.tag = tag
        setShape(shape: shape)
        setPos(position: position)
        setSize(size: size, scale: 1)
        setColor(color: color)
        setFill(fill: fill)
        setVisible(visible: visible)
    }
    
    func setShape(shape: ObjectShape) {
        properties.shape = shape
    }
    func setPos(x: Double, y: Double) {
        properties.x = x
        properties.y = y
    }
    func setPos(position: CGPoint) {
        properties.x = Double(position.x)
        properties.y = Double(position.y)
    }
    func setSize(w: Double, h: Double, scale: Double) {
        properties.w = w * scale
        properties.h = h * scale
    }
    func setSize(size: CGSize, scale: Double) {
        properties.w = Double(size.width) * scale
        properties.h = Double(size.height) * scale
    }
    func setColor(r: Int32, g: Int32, b: Int32, a: Int32 = 255) {
        properties.r = r
        properties.g = g
        properties.b = b
        properties.a = a
    }
    func setColor(color: UIColor) {
        var r: CGFloat = 0
        var g: CGFloat = 0
        var b: CGFloat = 0
        var a: CGFloat = 0
        color.getRed(&r, green: &g, blue: &b, alpha: &a)
        properties.r = Int32(r*255)
        properties.g = Int32(g*255)
        properties.b = Int32(b*255)
        properties.a = Int32(a*255)
    }
    func setFill(fill: Bool) {
        properties.fill = fill==true ? 1 : 0
    }
    func setVisible(visible: Bool) {
        properties.visible = visible==true ? 1 : 0
    }
    
    
    
    ///
    /// encode into a byte array that can be read by Java class TaskGraphObject
    ///
    /// test code to run on Matlab:
    ///    c1 = TaskGraphObject(4, 1, -1, 3, 10, 30, java.awt.Color(int32(255),int32(255),int32(255)), true, true);
    ///    b = c1.encode();
    ///    disp('transmitted array is:');
    ///    disp(encodeMatlabObject(b)')
    ///
    /// test code to run on Swift side:
    ///    let o1 = TaskGraphObj(tag: 4);
    ///    o1.setShape(shape: .RECTANGLE)
    ///    o1.setPos(x: -1, y: 3)
    ///    o1.setSize(w: 10, h: 30)
    ///    o1.setColor(r: 255, g: 255, b: 255)
    ///    o1.setFill(fill: true)
    ///    o1.setVisible(visible: true)
    ///    o1.encodeToBytes()
    func encodeToBytes() -> [Int8] {
        struct Packet {
            let tag : Int8
            let len : Int32
            let p : Properties
        }
        var packet = Packet(tag: 2, len: Int32(MemoryLayout<Properties>.size), p: properties)
        let d = Data(bytes: &packet, count: MemoryLayout<Packet>.size)
        let bytes_uint8 = [UInt8](d)
//alternative approach to [UInt8](d)
//print("length of d is: \(d.count)")
//var buf : [UInt8] = [UInt8](repeating: 0, count: d.count)
//d.copyBytes(to: &buf, count: d.count)
//print(buf)
        var bytes_int8 : [Int8] = bytes_uint8.map{Int8(bitPattern: $0)}
        
        //// weird corrections required! most likely because I do not fully understand how NSData works in Swift.
        //// revisit the code when your understanding improves
        //// CORRECTION 1
        //somehow Swift seems to treat packet.tag as Int32, even though it is explicitly defined
        //as Int8. this dumb behavior adds 3 additional bytes to the msg that must be removed. surprisingly,
        //these three bytes are added by Swift only when tag is followed by packet.len. after dropping len
        //from Packet, tag is represented by a single byte as it should be!
        let remove_bytes = bytes_int8.count - Int(packet.len+5)    //because we are including a struct within Packet, we have a little overhead in byte array
        if remove_bytes>0 {
            bytes_int8.removeSubrange(Range<Int>(NSRange(location: 1, length: 3))!)
        }
        //// CORRECTION 2
        //somehow Swift can put garbage in properties.shape because it cannot convert enum
        //ObjectShape to Int32 properly. I have to manually set the extra bytes to zero. it
        //is cosher for us as long as we have less than 255 shape types (that's too many, we have 4 now)
        bytes_int8[10] = 0
        bytes_int8[11] = 0
        bytes_int8[12] = 0
//print(properties)
//print("msg queue is: \(bytes_int8)")
        return bytes_int8
    }
}

