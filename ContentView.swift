//
//  ContentView.swift
//  CineMagic Pro
//
//  Single-file SwiftUI + AVFoundation + CoreImage + Vision implementation of a
//  cinematic camera app: live 4K/60fps preview, tap-to-focus, real-time
//  background isolation/blur (auto via Vision person segmentation, or manual
//  via a tap-anchored radial mask), cinematic LUT-style color grading, and
//  recording of the *processed* video stream via AVAssetWriter.
//
//  BUILD NOTES
//  -----------
//  This file assumes it lives inside a standard iOS App target. Two things
//  must be added at the project level (they cannot live inside a .swift file):
//
//  1. Info.plist keys:
//       NSCameraUsageDescription      – "CineMagic Pro needs camera access to record video."
//       NSMicrophoneUsageDescription  – "CineMagic Pro needs microphone access to record audio."
//       NSPhotoLibraryAddUsageDescription – "CineMagic Pro saves your cinematic clips to Photos."
//
//  2. This app requires a physical device (Metal + camera are unavailable in
//     the iOS Simulator). For unsigned CI builds (e.g. `xcodebuild build
//     CODE_SIGNING_ALLOWED=NO`), the app will compile fine but must be run on
//     device with a proper signing identity to actually use the camera.
//
//  KNOWN SIMPLIFICATIONS (called out inline below)
//  -------------------------------------------------
//  - Tap-to-focus maps view coordinates directly to AVCaptureDevice's
//    normalized focus point space. Production code should convert through
//    AVCaptureVideoPreviewLayer.captureDevicePointConverted(fromLayerPoint:);
//    since this app renders processed frames via a custom MTKView rather than
//    a preview layer, that convenience API isn't available, so a direct
//    normalized mapping is used instead.
//  - LUTs are approximated with composed CoreImage filters (color controls,
//    temperature/tint, sepia, vignette, noir) rather than loaded .cube LUT
//    files, since no LUT assets were provided.
//

import SwiftUI
import AVFoundation
import CoreImage
import CoreImage.CIFilterBuiltins
import Vision
import MetalKit
import Photos

// MARK: - LUT Style

enum LUTStyle: String, CaseIterable, Identifiable {
    case none        = "None"
    case cinema      = "Cinema"
    case warmSunset  = "Warm Sunset"
    case vintageFilm = "Vintage Film"
    case blockbuster = "Blockbuster"
    case moodyBlue   = "Moody Blue"
    case bwDrama     = "B&W Drama"

    var id: String { rawValue }
}

// MARK: - Isolation Mode

enum IsolationMode {
    case auto
    case manual
}

// MARK: - Frame Processor (CoreImage / Vision pipeline)

final class FrameProcessor {

    private let segmentationRequest: VNGeneratePersonSegmentationRequest = {
        let request = VNGeneratePersonSegmentationRequest()
        request.qualityLevel = .balanced
        request.outputPixelFormat = kCVPixelFormatType_OneComponent8
        return request
    }()

    /// Processes a single frame: isolates the subject (auto or manual),
    /// blurs the background by `blurIntensity`, then applies the selected LUT.
    /// `manualPoint` (if provided) is expected in CoreImage pixel space
    /// (origin bottom-left) matching `pixelBuffer`'s extent.
    func process(pixelBuffer: CVPixelBuffer,
                 mode: IsolationMode,
                 manualPoint: CGPoint?,
                 blurIntensity: Double,
                 lut: LUTStyle) -> CIImage? {

        let sourceImage = CIImage(cvPixelBuffer: pixelBuffer)
        let extent = sourceImage.extent

        let maskImage: CIImage?
        switch mode {
        case .auto:
            maskImage = personSegmentationMask(pixelBuffer: pixelBuffer, extent: extent)
        case .manual:
            maskImage = manualPoint.map { radialMask(around: $0, extent: extent) }
        }

        var working = sourceImage
        if blurIntensity > 0.01 {
            let blurFilter = CIFilter.gaussianBlur()
            blurFilter.inputImage = sourceImage.clampedToExtent()
            blurFilter.radius = Float(blurIntensity * 30.0)
            let blurred = (blurFilter.outputImage ?? sourceImage).cropped(to: extent)

            if let mask = maskImage {
                let blendFilter = CIFilter.blendWithMask()
                blendFilter.inputImage = sourceImage        // sharp subject
                blendFilter.backgroundImage = blurred       // blurred background
                blendFilter.maskImage = mask                // white = keep subject sharp
                working = (blendFilter.outputImage ?? sourceImage).cropped(to: extent)
            } else {
                working = blurred
            }
        }

        working = applyLUT(lut, to: working)
        return working
    }

    private func personSegmentationMask(pixelBuffer: CVPixelBuffer, extent: CGRect) -> CIImage? {
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        do {
            try handler.perform([segmentationRequest])
            guard let result = segmentationRequest.results?.first else { return nil }
            var maskImage = CIImage(cvPixelBuffer: result.pixelBuffer)
            let scaleX = extent.width / maskImage.extent.width
            let scaleY = extent.height / maskImage.extent.height
            maskImage = maskImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
            return maskImage
        } catch {
            return nil
        }
    }

    private func radialMask(around point: CGPoint, extent: CGRect) -> CIImage {
        let radialFilter = CIFilter.radialGradient()
        let radius = min(extent.width, extent.height) * 0.35
        radialFilter.center = point
        radialFilter.radius0 = Float(radius * 0.6)
        radialFilter.radius1 = Float(radius)
        radialFilter.color0 = CIColor.white
        radialFilter.color1 = CIColor.black
        return (radialFilter.outputImage ?? CIImage.empty()).cropped(to: extent)
    }

    private func applyLUT(_ lut: LUTStyle, to image: CIImage) -> CIImage {
        switch lut {
        case .none:
            return image

        case .cinema:
            let controls = CIFilter.colorControls()
            controls.inputImage = image
            controls.saturation = 1.05
            controls.contrast = 1.15
            controls.brightness = -0.02
            let tone = CIFilter.temperatureAndTint()
            tone.inputImage = controls.outputImage
            tone.neutral = CIVector(x: 6000, y: 0)
            tone.targetNeutral = CIVector(x: 5500, y: 0)
            return tone.outputImage ?? image

        case .warmSunset:
            let tone = CIFilter.temperatureAndTint()
            tone.inputImage = image
            tone.neutral = CIVector(x: 6500, y: 0)
            tone.targetNeutral = CIVector(x: 4400, y: 10)
            let controls = CIFilter.colorControls()
            controls.inputImage = tone.outputImage
            controls.saturation = 1.2
            controls.contrast = 1.05
            return controls.outputImage ?? image

        case .vintageFilm:
            let sepia = CIFilter.sepiaTone()
            sepia.inputImage = image
            sepia.intensity = 0.35
            let controls = CIFilter.colorControls()
            controls.inputImage = sepia.outputImage
            controls.contrast = 0.95
            controls.saturation = 0.85
            let vignette = CIFilter.vignette()
            vignette.inputImage = controls.outputImage
            vignette.intensity = 0.8
            vignette.radius = 1.8
            return vignette.outputImage ?? image

        case .blockbuster:
            let controls = CIFilter.colorControls()
            controls.inputImage = image
            controls.contrast = 1.25
            controls.saturation = 1.1
            let tone = CIFilter.temperatureAndTint()
            tone.inputImage = controls.outputImage
            tone.neutral = CIVector(x: 6500, y: 0)
            tone.targetNeutral = CIVector(x: 7200, y: -10)
            return tone.outputImage ?? image

        case .moodyBlue:
            let tone = CIFilter.temperatureAndTint()
            tone.inputImage = image
            tone.neutral = CIVector(x: 6500, y: 0)
            tone.targetNeutral = CIVector(x: 8500, y: 10)
            let controls = CIFilter.colorControls()
            controls.inputImage = tone.outputImage
            controls.saturation = 0.8
            controls.contrast = 1.1
            controls.brightness = -0.03
            return controls.outputImage ?? image

        case .bwDrama:
            let mono = CIFilter.photoEffectNoir()
            mono.inputImage = image
            let controls = CIFilter.colorControls()
            controls.inputImage = mono.outputImage
            controls.contrast = 1.2
            return controls.outputImage ?? image
        }
    }
}

// MARK: - Camera Manager

final class CameraManager: NSObject, ObservableObject {

    let session = AVCaptureSession()

    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "cinemagic.session.queue")
    private let processingQueue = DispatchQueue(label: "cinemagic.processing.queue")

    private var videoDevice: AVCaptureDevice?
    private let ciContext: CIContext
    private let frameProcessor = FrameProcessor()

    private var assetWriter: AVAssetWriter?
    private var writerVideoInput: AVAssetWriterInput?
    private var pixelBufferAdaptor: AVAssetWriterPixelBufferAdaptor?
    private var recordingSessionStarted = false
    private var outputURL: URL?

    @Published var isRecording = false
    @Published var isolationMode: IsolationMode = .auto
    @Published var manualIsolationPoint: CGPoint?   // normalized, 0...1
    @Published var blurIntensity: Double = 0.85
    @Published var selectedLUT: LUTStyle = .none
    @Published var latestFrame: CIImage?
    @Published var permissionGranted = false

    override init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is required and is unavailable on this device/simulator.")
        }
        self.ciContext = CIContext(mtlDevice: device)
        super.init()
    }

    // MARK: Permissions & session setup

    func requestPermissionAndConfigure() {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            DispatchQueue.main.async {
                self?.permissionGranted = granted
                if granted {
                    self?.configureSession()
                }
            }
        }
    }

    private func configureSession() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.session.beginConfiguration()
            self.session.sessionPreset = .inputPriority

            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                         for: .video,
                                                         position: .back) else {
                self.session.commitConfiguration()
                return
            }
            self.videoDevice = device

            // Prefer a 4K / 60fps-capable format; fall back to the device default.
            if let format = device.formats.first(where: { format in
                let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
                let is4K = dims.width >= 3840 && dims.height >= 2160
                let supports60 = format.videoSupportedFrameRateRanges.contains { $0.maxFrameRate >= 60 }
                return is4K && supports60
            }) {
                try? device.lockForConfiguration()
                device.activeFormat = format
                device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 60)
                device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 60)
                device.unlockForConfiguration()
            }

            do {
                let input = try AVCaptureDeviceInput(device: device)
                if self.session.canAddInput(input) {
                    self.session.addInput(input)
                }
            } catch {
                self.session.commitConfiguration()
                return
            }

            self.videoOutput.setSampleBufferDelegate(self, queue: self.processingQueue)
            self.videoOutput.alwaysDiscardsLateVideoFrames = true
            self.videoOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
            if self.session.canAddOutput(self.videoOutput) {
                self.session.addOutput(self.videoOutput)
            }
            self.videoOutput.connection(with: .video)?.videoOrientation = .portrait

            self.session.commitConfiguration()
            self.session.startRunning()
        }
    }

    // MARK: Focus / subject selection

    /// `normalizedPoint` is in 0...1 view space (origin top-left), from a tap
    /// on the live preview. See file header for the coordinate-mapping caveat.
    func focus(at normalizedPoint: CGPoint) {
        manualIsolationPoint = normalizedPoint

        sessionQueue.async { [weak self] in
            guard let self, let device = self.videoDevice else { return }
            do {
                try device.lockForConfiguration()
                if device.isFocusPointOfInterestSupported {
                    device.focusPointOfInterest = normalizedPoint
                    device.focusMode = .autoFocus
                }
                if device.isExposurePointOfInterestSupported {
                    device.exposurePointOfInterest = normalizedPoint
                    device.exposureMode = .autoExpose
                }
                device.unlockForConfiguration()
            } catch {
                // Ignore focus configuration failures; camera keeps running.
            }
        }
    }

    // MARK: Recording (writes the *processed* frames, not raw camera output)

    func startRecording() {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("mov")
        outputURL = url
        recordingSessionStarted = false

        guard let writer = try? AVAssetWriter(outputURL: url, fileType: .mov) else { return }

        let outputSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: 2160,
            AVVideoHeightKey: 3840
        ]
        let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: outputSettings)
        videoInput.expectsMediaDataInRealTime = true

        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: 2160,
            kCVPixelBufferHeightKey as String: 3840
        ]
        let adaptor = AVAssetWriterPixelBufferAdaptor(assetWriterInput: videoInput,
                                                        sourcePixelBufferAttributes: attributes)

        if writer.canAdd(videoInput) {
            writer.add(videoInput)
        }

        assetWriter = writer
        writerVideoInput = videoInput
        pixelBufferAdaptor = adaptor

        writer.startWriting()
        DispatchQueue.main.async { self.isRecording = true }
    }

    func stopRecording(completion: @escaping (URL?) -> Void) {
        guard let writer = assetWriter, writer.status == .writing else {
            DispatchQueue.main.async { self.isRecording = false }
            completion(nil)
            return
        }
        writerVideoInput?.markAsFinished()
        writer.finishWriting { [weak self] in
            DispatchQueue.main.async {
                self?.isRecording = false
                completion(self?.outputURL)
            }
        }
    }

    private func appendProcessedFrame(_ image: CIImage, presentationTime: CMTime) {
        guard isRecording,
              let writer = assetWriter,
              let adaptor = pixelBufferAdaptor,
              let pool = adaptor.pixelBufferPool,
              let input = writerVideoInput,
              input.isReadyForMoreMediaData else { return }

        if !recordingSessionStarted {
            writer.startSession(atSourceTime: presentationTime)
            recordingSessionStarted = true
        }

        var pixelBufferOut: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pixelBufferOut)
        guard let outBuffer = pixelBufferOut else { return }

        ciContext.render(image, to: outBuffer)
        adaptor.append(outBuffer, withPresentationTime: presentationTime)
    }

    private func imagePoint(fromNormalized normalized: CGPoint, extent: CGRect) -> CGPoint {
        // CoreImage's coordinate origin is bottom-left; UIKit's is top-left.
        CGPoint(x: normalized.x * extent.width,
                y: (1 - normalized.y) * extent.height)
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                        didOutput sampleBuffer: CMSampleBuffer,
                        from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let presentationTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)

        let mode = isolationMode
        let blur = blurIntensity
        let lut = selectedLUT
        let extent = CIImage(cvPixelBuffer: pixelBuffer).extent
        let manualPoint = manualIsolationPoint.map { imagePoint(fromNormalized: $0, extent: extent) }

        guard let processed = frameProcessor.process(pixelBuffer: pixelBuffer,
                                                       mode: mode,
                                                       manualPoint: manualPoint,
                                                       blurIntensity: blur,
                                                       lut: lut) else { return }

        DispatchQueue.main.async { [weak self] in
            self?.latestFrame = processed
        }

        appendProcessedFrame(processed, presentationTime: presentationTime)
    }
}

// MARK: - Metal-backed live preview (renders the processed CIImage each frame)

final class TapableMTKView: MTKView {}

struct MetalPreviewView: UIViewRepresentable {
    @ObservedObject var cameraManager: CameraManager

    func makeCoordinator() -> Coordinator {
        Coordinator(cameraManager: cameraManager)
    }

    func makeUIView(context: Context) -> TapableMTKView {
        let device = MTLCreateSystemDefaultDevice()!
        let view = TapableMTKView(frame: .zero, device: device)
        view.delegate = context.coordinator
        view.framebufferOnly = false
        view.isPaused = false
        view.enableSetNeedsDisplay = false
        view.preferredFramesPerSecond = 60
        view.backgroundColor = .black

        let tap = UITapGestureRecognizer(target: context.coordinator,
                                          action: #selector(Coordinator.handleTap(_:)))
        view.addGestureRecognizer(tap)

        return view
    }

    func updateUIView(_ uiView: TapableMTKView, context: Context) {}

    final class Coordinator: NSObject, MTKViewDelegate {
        let cameraManager: CameraManager
        let commandQueue: MTLCommandQueue
        let ciContext: CIContext

        init(cameraManager: CameraManager) {
            self.cameraManager = cameraManager
            let device = MTLCreateSystemDefaultDevice()!
            self.commandQueue = device.makeCommandQueue()!
            self.ciContext = CIContext(mtlDevice: device)
            super.init()
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

        func draw(in view: MTKView) {
            guard let drawable = view.currentDrawable,
                  let image = cameraManager.latestFrame,
                  let commandBuffer = commandQueue.makeCommandBuffer() else { return }

            let drawableSize = view.drawableSize
            guard drawableSize.width > 0, drawableSize.height > 0, image.extent.width > 0 else { return }

            // Aspect-fill the processed image into the drawable.
            let scale = max(drawableSize.width / image.extent.width,
                             drawableSize.height / image.extent.height)
            let scaled = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            let originX = (drawableSize.width - scaled.extent.width) / 2
            let originY = (drawableSize.height - scaled.extent.height) / 2
            let centered = scaled.transformed(by: CGAffineTransform(translationX: originX, y: originY))

            ciContext.render(centered,
                              to: drawable.texture,
                              commandBuffer: commandBuffer,
                              bounds: CGRect(origin: .zero, size: drawableSize),
                              colorSpace: CGColorSpaceCreateDeviceRGB())

            commandBuffer.present(drawable)
            commandBuffer.commit()
        }

        @objc func handleTap(_ gesture: UITapGestureRecognizer) {
            guard let view = gesture.view, view.bounds.width > 0, view.bounds.height > 0 else { return }
            let location = gesture.location(in: view)
            let normalized = CGPoint(x: location.x / view.bounds.width,
                                      y: location.y / view.bounds.height)
            cameraManager.focus(at: normalized)
        }
    }
}

// MARK: - Main UI

struct ContentView: View {
    @StateObject private var cameraManager = CameraManager()
    @State private var showLUTPicker = false
    @State private var showSavedAlert = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if cameraManager.permissionGranted {
                MetalPreviewView(cameraManager: cameraManager)
                    .ignoresSafeArea()
            } else {
                permissionView
            }

            VStack(spacing: 0) {
                topBar
                    .padding(.top, 8)
                Spacer()
                if showLUTPicker {
                    lutPicker
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
                bottomControls
            }
        }
        .statusBarHidden(true)
        .preferredColorScheme(.dark)
        .onAppear { cameraManager.requestPermissionAndConfigure() }
        .alert("Clip Saved", isPresented: $showSavedAlert) {
            Button("OK", role: .cancel) {}
        } message: {
            Text("Your cinematic clip was saved to Photos.")
        }
    }

    private var permissionView: some View {
        VStack(spacing: 16) {
            Image(systemName: "camera.fill")
                .font(.system(size: 44))
                .foregroundStyle(.white.opacity(0.7))
            Text("Camera access is required for CineMagic Pro.")
                .foregroundStyle(.white.opacity(0.8))
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
        }
    }

    private var topBar: some View {
        HStack {
            Label("4K 60fps", systemImage: "camera.aperture")
                .font(.caption.weight(.semibold))
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(.ultraThinMaterial, in: Capsule())

            Spacer()

            Picker("", selection: $cameraManager.isolationMode) {
                Text("Auto").tag(IsolationMode.auto)
                Text("Manual").tag(IsolationMode.manual)
            }
            .pickerStyle(.segmented)
            .frame(width: 160)

            Spacer()

            if cameraManager.isolationMode == .manual {
                Text("Tap subject")
                    .font(.caption2.weight(.medium))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(.ultraThinMaterial, in: Capsule())
            } else {
                Color.clear.frame(width: 90)
            }
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 16)
    }

    private var bottomControls: some View {
        VStack(spacing: 14) {
            isolationSlider

            HStack(spacing: 28) {
                controlButton(icon: "camera.aperture", label: "LENS") {}
                controlButton(icon: "viewfinder", label: "FOCUS") {}

                recordButton

                controlButton(icon: "paintpalette", label: "LUTs") {
                    withAnimation { showLUTPicker.toggle() }
                }
                controlButton(icon: "gauge.with.dots.needle.bottom.50percent", label: "ISO") {}
            }
        }
        .padding(.horizontal, 20)
        .padding(.bottom, 20)
    }

    private var isolationSlider: some View {
        VStack(spacing: 6) {
            HStack {
                Text("Isolation Intensity")
                Spacer()
                Text("\(Int(cameraManager.blurIntensity * 100))%")
            }
            .font(.caption.weight(.medium))
            .foregroundStyle(.white.opacity(0.85))

            Slider(value: $cameraManager.blurIntensity, in: 0...1)
                .tint(.orange)
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
        .padding(.horizontal, 20)
    }

    private var recordButton: some View {
        Button(action: toggleRecording) {
            ZStack {
                Circle()
                    .stroke(Color.white.opacity(0.85), lineWidth: 4)
                    .frame(width: 74, height: 74)

                RoundedRectangle(cornerRadius: cameraManager.isRecording ? 8 : 30)
                    .fill(Color.red)
                    .frame(width: cameraManager.isRecording ? 30 : 60,
                           height: cameraManager.isRecording ? 30 : 60)
                    .animation(.easeInOut(duration: 0.2), value: cameraManager.isRecording)
            }
        }
    }

    private func controlButton(icon: String, label: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 18))
                Text(label)
                    .font(.system(size: 10, weight: .semibold))
            }
            .foregroundStyle(.white)
            .frame(width: 52, height: 52)
            .background(.ultraThinMaterial, in: Circle())
        }
    }

    private var lutPicker: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 12) {
                ForEach(LUTStyle.allCases) { lut in
                    Button {
                        cameraManager.selectedLUT = lut
                    } label: {
                        VStack(spacing: 6) {
                            RoundedRectangle(cornerRadius: 10)
                                .fill(LinearGradient(colors: swatchColors(for: lut),
                                                      startPoint: .topLeading,
                                                      endPoint: .bottomTrailing))
                                .frame(width: 64, height: 64)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 10)
                                        .stroke(cameraManager.selectedLUT == lut ? Color.orange : Color.clear,
                                                lineWidth: 3)
                                )
                            Text(lut.rawValue)
                                .font(.system(size: 10, weight: .medium))
                                .foregroundStyle(.white.opacity(0.85))
                        }
                    }
                }
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 10)
        }
        .background(.ultraThinMaterial)
    }

    private func swatchColors(for lut: LUTStyle) -> [Color] {
        switch lut {
        case .none:        return [.gray, .gray.opacity(0.6)]
        case .cinema:       return [.blue, .black]
        case .warmSunset:   return [.orange, .pink]
        case .vintageFilm:  return [.brown, .yellow]
        case .blockbuster:  return [.indigo, .black]
        case .moodyBlue:    return [.blue, .indigo]
        case .bwDrama:      return [.gray, .black]
        }
    }

    private func toggleRecording() {
        if cameraManager.isRecording {
            cameraManager.stopRecording { url in
                guard let url else { return }
                saveToPhotos(url: url)
            }
        } else {
            cameraManager.startRecording()
        }
    }

    private func saveToPhotos(url: URL) {
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else { return }
            PHPhotoLibrary.shared().performChanges({
                PHAssetCreationRequest.creationRequestForAssetFromVideo(atFileURL: url)
            }) { success, _ in
                DispatchQueue.main.async {
                    if success { showSavedAlert = true }
                }
            }
        }
    }
}

// MARK: - App Entry Point

@main
struct CineMagicProApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
