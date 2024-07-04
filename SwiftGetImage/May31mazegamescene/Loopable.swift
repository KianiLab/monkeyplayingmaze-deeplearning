//
//  Loopable.swift
//  May31mazegamescene
//
//  
//


import Foundation
import UIKit


protocol Loopable {
    func getPropertyList() throws -> [String: Any]
    func toString() throws -> String
}

extension Loopable {
    
    //getPropertyList()
    func getPropertyList() throws -> [String: Any] {
        var result: [String: Any] = [:]
        //mirror the struct
        let mirror = Mirror(reflecting: self)
        // Optional check to make sure we're iterating over a struct or class
        guard let style = mirror.displayStyle, style == .struct || style == .class else {
            throw NSError()
        }
        //gerenate the property list of the form [property_name : property_value]
        for (property, value) in mirror.children {
            guard let property = property else {
                continue
            }
            
            result[property] = value
        }
        return result
    }
    
    
    //toString()
    func toString() throws -> String {
        var str : String = ""
        //mirror the struct
        let mirror = Mirror(reflecting: self)
        // Optional check to make sure we're iterating over a struct or class
        guard let style = mirror.displayStyle, style == .struct || style == .class else {
            throw NSError()
        }
        //gerenate the property list of the form [property_name : property_value]
        for (property, value) in mirror.children {
            guard let property = property else {
                continue
            }
            let mirror_child = Mirror(reflecting: value)
            if let style_child = mirror_child.displayStyle, style_child == .struct || style_child == .class {
                str += "\(property): { \n"
                if let v = value as? Loopable {
                    do {
                        str += try v.toString()
                    } catch let error as NSError {
                        str += "\(error.localizedDescription)"
                    }
                } else {
                    str += "\(value) \n"
                }
                str += "} \n"
            } else {
                //take spsecial measure to remove floating point imprecision by rounding to 1e-6
                if value is Float {
                    if var v = value as? Float {
                        v = Float(round(Double(v)*1e6)/1e6)
                        str += "\(property): \(v) \n"
                    }
                } else if value is [Float] {
                    if var v = value as? [Float] {
                        for i in 0..<v.count {
                            v[i] = Float(round(Double(v[i])*1e6)/1e6)
                        }
                        str += "\(property): \(v) \n"
                    }
                } else if value is CGFloat {
                    if var v = value as? CGFloat {
                        v = CGFloat(round(Double(v)*1e6)/1e6)
                        str += "\(property): \(v) \n"
                    }
                } else if value is [CGFloat] {
                    if var v = value as? [CGFloat] {
                        for i in 0..<v.count {
                            v[i] = CGFloat(round(Double(v[i])*1e6)/1e6)
                        }
                        str += "\(property): \(v) \n"
                    }
                } else {
                    //no special measure is needed for non-float variables
                    str += "\(property): \(value) \n"
                }
            }
        }
        return str
    }
    
    
    // toString(exclude_list)
    func toString(exclude : [String]) throws -> String {
        var str : String = ""
        //mirror the struct
        let mirror = Mirror(reflecting: self)
        // Optional check to make sure we're iterating over a struct or class
        guard let style = mirror.displayStyle, style == .struct || style == .class else {
            throw NSError()
        }
        
        let dict = mirror.children.filter { (child) -> Bool in
            guard let label = child.label else {
                return false
            }
            return (exclude.contains(label) == false)
        }
        
        //gerenate the property list of the form [property_name : property_value]
        for (property, value) in dict {
            guard let property = property else {
                continue
            }
            let mirror_child = Mirror(reflecting: value)
            if let style_child = mirror_child.displayStyle, (style_child == .struct || style_child == .class) {
                str += "\(property): { \n"
                if let v = value as? Loopable {
                    do {
                        str += try v.toString(exclude: exclude)
                    } catch let error as NSError {
                        str += "\(error.localizedDescription)"
                    }
                } else {
                    str += "\(value) \n"
                }
                str += "} \n"
            } else {
                str += "\(property): \(value) \n"
            }
        }
        return str
    }
}
