# M2 Multi-Modal Ghost Vision

A real-time, hardware-accelerated sensor fusion pipeline built from scratch in C++ and Objective-C++ for Apple Silicon. 

This project fuses optical computer vision (YOLOv8) with invisible radio frequency (RF) spectrum scanning (Wi-Fi), and feeds both data streams into a locally running Large Language Model to generate real-time, natural language scene descriptions.

## System Architecture
1. **The Optical Eye (CoreML):** Uses the `onnxruntime` C++ API with the CoreML Execution Provider to run YOLOv8n object detection natively on the M2 Neural Engine.
2. **The RF Spectrum Scanner (CoreWLAN):** A background Objective-C++ thread asynchronously polls the macOS native Wi-Fi hardware, sniffing raw RSSI signal strengths from nearby devices without blocking the video thread.
3. **The Thermal Renderer (OpenCV):** Transforms invisible RF data into a spatial 2D grid, applying bicubic interpolation and a Jet colormap alpha-blended over the live camera feed to visualize the local Wi-Fi environment.
4. **The Cognitive Brain (llama.cpp):** Integrates the `llama.cpp` v0.9 API directly into the pipeline, utilizing the Metal GPU to run a 4-bit quantized Qwen-1.5B LLM. It ingests both the YOLO bounding boxes and the RF hotspot locations to stream real-time, unified natural-language scene reports.

## Core Competencies Demonstrated
* C++17 memory management and `std::thread` synchronization.
* Objective-C++ bridging for native macOS hardware access.
* Apple Silicon hardware acceleration (CoreML + Metal GPU).
* Edge-AI inference and multi-modal sensor fusion.

## Build Instructions
Requires OpenCV, ONNX Runtime, and llama.cpp. Compiled via CMake.
