import UIKit
import SpriteKit
import GameplayKit

class GameViewController: UIViewController {
    var skView: SKView!
    var jsonFilenames: [String] = []

    override func viewDidLoad() {
        super.viewDidLoad()

        // Setup SKView and GameScene
        skView = SKView(frame: self.view.bounds)
        self.view.addSubview(skView)
        
        // Load JSON filenames from the app bundle
        loadJSONFilenames()

        // Generate images for all JSON files
        generateImagesForAllJSONFiles()
    }

    // Load JSON filenames from the app bundle
    func loadJSONFilenames() {
        guard let resourcePath = Bundle.main.resourcePath else {
            print("Resource path not found.")
            return
        }

        do {
            let items = try FileManager.default.contentsOfDirectory(atPath: resourcePath)
            jsonFilenames = items.filter { $0.hasSuffix(".json") }
            jsonFilenames.sort() // Sort filenames if needed
            print("Found JSON files: \(jsonFilenames)")
        } catch {
            print("Failed to load contents of directory: \(error)")
        }
    }

//    // Generate images for all JSON files
//    func generateImagesForAllJSONFiles() {
//        for jsonFilename in jsonFilenames {
//            generateImage(for: jsonFilename)
//        }
//    }
//
//    // Generate image for a specific JSON file
//    func generateImage(for jsonFilename: String) {
//        // Create and present the scene
//        let scene = GameScene(size: self.view.bounds.size, jsonFilename: jsonFilename)
//        skView.presentScene(scene)
//
//        // Wait a short period to ensure the scene is fully rendered
//        DispatchQueue.main.asyncAfter(deadline: .now() +  1.5 ) { // Adjust delay time as needed
//            let screenshot = self.captureScreen()
//            self.saveImage(screenshot, toPath: "/Users/xinglanzhao/Desktop/June20images/\(jsonFilename).png")
//        }
//    }
    func generateImagesForAllJSONFiles() {
        generateImage(index: 0)
    }

    // Recursive function to handle one file at a time
    func generateImage(index: Int) {
        //base case 
        if index >= jsonFilenames.count {
            print("All images have been generated.")
            return
        }

        let jsonFilename = jsonFilenames[index]
        let scene = GameScene(size: self.view.bounds.size, jsonFilename: jsonFilename)
        skView.presentScene(scene)

        // Wait a short period to ensure the scene is fully rendered
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) { // Adjust delay time as needed for scene complexity
//            let screenshot = self.captureScreen()
//            let savePath = "/Users/xinglanzhao/Desktop/June19/image/\(jsonFilename).png"
//            self.saveImage(screenshot, toPath: savePath)
//            print("Image for \(jsonFilename) saved.")
            
            let screenshot = self.captureScreen()
                   
                   // Split the filename to remove the extension and use only the base name
                   let baseFilename = jsonFilename.components(separatedBy: ".").first ?? "default"
                   let savePath = "/Users/xinglanzhao/Desktop/June19/image/\(baseFilename).png"
                   
                   self.saveImage(screenshot, toPath: savePath)
                   print("Image for \(baseFilename) saved.")

            // Proceed to the next file
            self.generateImage(index: index + 1)
        }
    }

    func captureScreen() -> UIImage {
        // Begin an image context with scale factor set to 1.0 to ignore device scale
        UIGraphicsBeginImageContextWithOptions(skView.bounds.size, false, 1.0)
        skView.drawHierarchy(in: skView.bounds, afterScreenUpdates: true)
        guard let image = UIGraphicsGetImageFromCurrentImageContext() else {
            fatalError("Could not capture the screen")
        }
        UIGraphicsEndImageContext()
        return image
    }

    private func saveImage(_ image: UIImage, toPath path: String) {
        guard let data = image.pngData() else {
            print("Failed to generate PNG data.")
            return
        }

        let url = URL(fileURLWithPath: path)
        do {
            try data.write(to: url)
            print("Saved image to \(url.path)")
        } catch {
            print("Failed to save image: \(error)")
        }
    }

    override var shouldAutorotate: Bool {
        return true
    }

    override var supportedInterfaceOrientations: UIInterfaceOrientationMask {
        if UIDevice.current.userInterfaceIdiom == .phone {
            return .landscape
        } else {
            return .landscape
        }
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Release any cached data, images, etc that aren't in use.
    }

    override var prefersStatusBarHidden: Bool {
        return true
    }
}

