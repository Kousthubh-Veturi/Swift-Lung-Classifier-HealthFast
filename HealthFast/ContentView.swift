//
//  ContentView.swift
//  HealthFast
//
//  Created by Kousthubh Veturi on 4/9/24.
//

import SwiftUI
import SwiftData
import SQLite

struct ContentView: SwiftUI.View {
    @StateObject private var auth = UserAuthentication()
    @State private var username: String = ""
    @State private var password: String = ""
    @State private var showingSignUp = false
    
    var body: some SwiftUI.View {
        HStack {
            // Left sidebar area for the buttons
            VStack {
                TextField("Username", text: $username)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .padding()
                SecureField("Password", text: $password)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .padding()
                
                Button("Login") {
                    auth.login(username: username, password: password)
                }
                .buttonStyle(PlainButtonStyle())
                .padding()
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(10)
                
                Button("Sign Up") {
                    // Present sign up view or form
                    showingSignUp = true
                }
                .buttonStyle(PlainButtonStyle())
                .padding()
                .background(Color.green)
                .foregroundColor(.white)
                .cornerRadius(10)
                .sheet(isPresented: $showingSignUp) {
                    SignUpView(auth: auth)
                }

                Button("Use Without Login") {
                    auth.useWithoutLogin()
                }
                .buttonStyle(PlainButtonStyle())
                .padding()
                .background(Color.orange)
                .foregroundColor(.white)
                .cornerRadius(10)
                
                Spacer()
            }
            .frame(width: 200)
            .padding(.top, 22)
            
            Divider()
            
            // Main content area that depends on the login state
            if auth.isLoggedIn {
                MainContentView()
                    
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
import SwiftUI
import CoreML
import Vision

struct MainContentView: SwiftUI.View {
    @State private var selectedImage: NSImage?
    @State private var showingImagePicker = false
    @State private var processedImage: CGImage?
    @State private var runTestEnabled = false
    @State private var modelOutput: String?
    //@State private var floatArray: Array<Float32>

    var body: some SwiftUI.View {
        VStack {
            if let selectedImage = selectedImage {
                Image(nsImage: selectedImage)
                    .resizable()
                    .scaledToFit()
            }

            Button("Upload Image") {
                showingImagePicker = true
                print(showingImagePicker)
                
            }

            if runTestEnabled {
                Button("Run Test") {
                    runMLModel()
                }
            }

            // Show the model output
            if let modelOutput = modelOutput {
                Text("Model Output: \(modelOutput)")
                 
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .fileImporter(isPresented: $showingImagePicker, allowedContentTypes: [.png, .jpeg]) { result in
            switch result {
            case .success(let url):
                print("File URL: \(url.path)") // Log the path to see if it's correct
                /*if let url = URL(string: "/Users/kousthubhveturi/Downloads/Screenshot 2024-04-14 at 2.30.39 PM.png"), // Change this path to a known image path
                   let image = NSImage(contentsOf: url) {
                    print("Manual load successful")
                } else {
                    print("Manual load failed")
                }*/
                if let image = NSImage(contentsOf: url) {
                    selectedImage = image
                    runTestEnabled = true
                    print("Image loaded successfully")
                } else {
                    print("Failed to load image from URL: \(url)")
                }
            case .failure(let error):
                print("Image loading failed with error: \(error)")
            }
        }
    }
    func convert(mlMultiArray: MLMultiArray) -> [Float] {
        let count = mlMultiArray.count
        let pointer = mlMultiArray.dataPointer.assumingMemoryBound(to: Float.self)
        let buffer = UnsafeBufferPointer(start: pointer, count: count)
        return Array(buffer)
    }
    private func runMLModel() {
        print("Attempting to resize image")
        guard let resizedImage = selectedImage?.resized(to: CGSize(width: 224, height: 224)) else {
            print("Failed to resize image")
            return
        }
        print("Successfully resized image")
        
        print("Attempting to convert image to buffer")
        guard let buffer = resizedImage.toBuffer() else {
            print("Failed to convert image to buffer")
            return
        }
        print("Successfully converted image to buffer")
        
        // Assuming 'lungimagesets' is the name of the ML model class generated by Core ML
        do {
            let model = try lungimagesets()
            print("got here2")
            let prediction = try model.prediction(conv2d_input: buffer)
            let outputss = prediction.Identity
            let modelOutput = "The result is: \(convert(mlMultiArray: outputss))"
            print("The results are: \(modelOutput)")
            
            
        } catch {
            print(error.localizedDescription)
            modelOutput = "Error running model."
        }
    }
}


extension NSImage {
    func resized(to newSize: CGSize) -> NSImage? {
            guard let tiffData = self.tiffRepresentation,
                  let bitmapImage = NSBitmapImageRep(data: tiffData),
                  let cgImage = bitmapImage.cgImage else {
                print("Failed to get CGImage")
                return nil
            }

            let width = Int(newSize.width)
            let height = Int(newSize.height)
            let bitsPerComponent = 8
            let bytesPerPixel = 4
            let colorSpace = CGColorSpaceCreateDeviceRGB()
            let bitmapInfo = CGImageAlphaInfo.premultipliedFirst.rawValue
            guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: bitsPerComponent, bytesPerRow: width * bytesPerPixel, space: colorSpace, bitmapInfo: bitmapInfo) else {
                print("Failed to create CGContext")
                return nil
            }

            context.interpolationQuality = .high
            context.draw(cgImage, in: CGRect(origin: .zero, size: newSize))

            guard let scaledCgImage = context.makeImage() else {
                print("Failed to create scaled CGImage")
                return nil
            }

            return NSImage(cgImage: scaledCgImage, size: newSize)
    }

    func toBuffer() -> CVPixelBuffer? {
        guard let data = self.tiffRepresentation,
        let bitmapRep = NSBitmapImageRep(data: data) else { return nil }

        let width = 224
        let height = 224

        var pixelBuffer: CVPixelBuffer?
        let attributes: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferWidthKey: NSNumber(value: width),
            kCVPixelBufferHeightKey: NSNumber(value: height),
            kCVPixelBufferPixelFormatTypeKey: NSNumber(value: Int32(kCVPixelFormatType_32ARGB))
        ]

        let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(width), Int(height), kCVPixelFormatType_32ARGB, attributes as CFDictionary, &pixelBuffer)

        guard status == kCVReturnSuccess, let imageBuffer = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(imageBuffer, [])
        let pixelData = CVPixelBufferGetBaseAddress(imageBuffer)

        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixelData,
                                        width: Int(width),
                                        height: Int(height),
                                        bitsPerComponent: 8,
                                        bytesPerRow: CVPixelBufferGetBytesPerRow(imageBuffer),
                                        space: rgbColorSpace,
                                        bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)

        context?.clear(CGRect(x: 0, y: 0, width: width, height: height))
        context?.draw(bitmapRep.cgImage!, in: CGRect(x: 0, y: 0, width: width, height: height))
        CVPixelBufferUnlockBaseAddress(imageBuffer, [])

        return imageBuffer
        
    }
}

struct SignUpView: SwiftUI.View {
    @ObservedObject var auth: UserAuthentication
    @State private var username: String = ""
    @State private var password: String = ""
    @State private var confirmPassword: String = ""
    @Environment(\.presentationMode) var presentationMode
    
    var body: some SwiftUI.View {
        VStack(spacing: 20) {
            Text("Sign Up")
                .font(.largeTitle)
            TextField("Username", text: $username)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding()
            SecureField("Password", text: $password)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding()
            SecureField("Confirm Password", text: $confirmPassword)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding()
            
            Button("Create Account") {
                auth.signUp(username: username, password: password, confirmPassword: confirmPassword)
                presentationMode.wrappedValue.dismiss()
            }
            .disabled(username.isEmpty || password.isEmpty || password != confirmPassword)
        }
        .padding()
        .frame(width: 300, height: 300)
    }
}


class UserAuthentication: ObservableObject {
    @Published var isLoggedIn = false
    private var database: Connection?

    init() {
        do{
            let fileManager = FileManager.default
            let documentsDirectory = try fileManager.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
            let dbPath = documentsDirectory.appendingPathComponent("HealthFastApp.sqlite3").path
            if !fileManager.fileExists(atPath: dbPath) {
                database = try Connection(dbPath)
                try createTables()
            } else {
                database = try Connection(dbPath)
            }
        }catch{
            print("database connection error")
        }
    }
    private func createTables() throws {
        let users = Table("users")
        let id = Expression<Int64>("id")
        let username = Expression<String>("username")
        let password = Expression<String>("password")
            
        try database?.run(users.create { t in
            t.column(id, primaryKey: .autoincrement)
            t.column(username, unique: true)
            t.column(password)
        })
    }
        
    func login(username: String, password: String) {
        let users = Table("users")
        let usernameColumn = Expression<String>("username")
        let passwordColumn = Expression<String>("password")
        
        let query = users.filter(usernameColumn == username && passwordColumn == password)
        
        do {
            let user = try database?.pluck(query)
            if user != nil {
                DispatchQueue.main.async {
                    self.isLoggedIn = true
                }
            } else {
                print("incorrect password")
            }
        } catch {
            print("Login failed: \(error)")
        }
    }
    

    func useWithoutLogin() {
        isLoggedIn = true
    }
    

}

extension UserAuthentication {
    func signUp(username: String, password: String, confirmPassword: String) {
        guard password == confirmPassword else {
            // Handle password mismatch
            print("Passwords do not match")
            return
        }
        
        let users = Table("users")
        let usernameColumn = Expression<String>("username")
        let passwordColumn = Expression<String>("password")
        
        let insert = users.insert(usernameColumn <- username, passwordColumn <- password)
        
        do {
            try database?.run(insert)
            print("User successfully signed up")
        } catch {
            print("Sign up failed: \(error)")
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some SwiftUI.View {
        ContentView()
    }
}
