// =============================================================================
// main.mm : YOLOv8n + llama.cpp Real-Time Vision & Language Pipeline
//           ONNX Runtime (CoreML EP) + llama.cpp (Metal GPU) + OpenCV
//           + RF Thermal Ghost Vision (Phase 2: live Wi-Fi scan via CoreWLAN)
//
// Objective-C++ (compiled with -fobjc-arc, see CMakeLists.txt).
// =============================================================================

// C++ / third-party headers first, Apple headers LAST: Foundation defines
// macros (MIN, MAX, YES, NO) that can collide with OpenCV and the STL.
#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include "llama.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <sstream>

#import <Foundation/Foundation.h>
#import <CoreWLAN/CoreWLAN.h>

struct Detection { float x; float y; float width; float height; int class_id; float confidence; };
struct LetterboxInfo { float scale; int pad_x; int pad_y; };

static const char* kClassNames[80] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
};

// =============================================================================
// RF Thermal Ghost Vision: shared grid + background CoreWLAN scanner thread
// =============================================================================

constexpr int    kGridCols      = 16;      // 16x12 matches a 4:3 webcam frame
constexpr int    kGridRows      = 12;
constexpr float  kRssiMin       = -90.0f;  // dBm shown as "cold" (blue)
constexpr float  kRssiMax       = -30.0f;  // dBm shown as "hot"  (red)
constexpr float  kStrongRssi    = -55.0f;  // peak above this counts as "strong"
constexpr double kThermalAlpha  = 0.30;    // overlay opacity (30%)

// A plain copy of the grid: kGridRows x kGridCols RSSI values in dBm.
using RfSnapshot = std::array<std::array<float, kGridCols>, kGridRows>;

// Owns the shared grid, its mutex, and the background thread that scans Wi-Fi.
// (Name kept from Phase 1 so the call sites in main() don't change.)
//
// THREAD SAFETY (unchanged from Phase 1):
//   * grid_ is shared between the video thread (reader) and the worker thread
//     (writer); every access goes through mtx_.
//   * snapshot() returns a COPY made under the lock.
//   * The worker performs the slow scan and builds the new grid OUTSIDE the
//     lock, and only locks to swap the finished grid in.
//   * The destructor sets stop_, notifies the condition variable and joins.
//     NOTE: a CoreWLAN scan cannot be interrupted, so if the app exits in the
//     middle of a scan the join waits for it to finish (up to ~3 s).
class RfSimulator {
public:
    RfSimulator() {
        for (auto& row : grid_) row.fill(kRssiMin);  // all "cold" until the first scan finishes
        worker_ = std::thread(&RfSimulator::run, this);  // started last
    }

    ~RfSimulator() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stop_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    RfSimulator(const RfSimulator&)            = delete;
    RfSimulator& operator=(const RfSimulator&) = delete;

    RfSnapshot snapshot() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return grid_;
    }

private:
    // One access point reduced to what the grid needs.
    struct ApReading { int x; int y; float rssi; };

    // 64-bit FNV-1a: a hash with a FIXED, documented definition. (std::hash is
    // allowed to differ between runs/implementations, which would move the
    // blobs around; FNV-1a gives the same cell for the same MAC every time.)
    static uint64_t fnv1a(const void* data, size_t n) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        uint64_t h = 1469598103934665603ULL;
        for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
        // murmur3 finalizer: spreads entropy into the bits we use for x and y
        h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
        return h;
    }

    // Turns the live APs into a grid. Each AP draws a Gaussian blob centered
    // on its hashed cell; blobs are summed on top of the "cold" floor.
    //
    // amplitude = (rssi - kRssiMin) dB, so an isolated AP at -45 dBm peaks at
    // exactly -45 dBm in its own cell. Stronger APs also get a slightly wider blob.
    static RfSnapshot render_blobs(const std::vector<ApReading>& aps) {
        RfSnapshot g{};
        for (auto& row : g) row.fill(kRssiMin);

        for (const ApReading& ap : aps) {
            const float amp   = std::max(0.0f, ap.rssi - kRssiMin);
            const float sigma = 1.3f + 0.9f * (amp / (kRssiMax - kRssiMin));
            const float inv2s2 = 1.0f / (2.0f * sigma * sigma);
            for (int r = 0; r < kGridRows; ++r) {
                for (int c = 0; c < kGridCols; ++c) {
                    const float dx = static_cast<float>(c - ap.x);
                    const float dy = static_cast<float>(r - ap.y);
                    g[r][c] += amp * std::exp(-(dx * dx + dy * dy) * inv2s2);
                }
            }
        }
        for (auto& row : g)
            for (float& v : row) v = std::clamp(v, kRssiMin, kRssiMax);
        return g;
    }

    // One blocking CoreWLAN scan (~2-3 s). Runs on the worker thread only.
    // Returns false (leaving the previous grid in place) if the scan failed.
    bool scan_once(RfSnapshot& out) {
        // std::thread is not an NSThread with its own autorelease pool, so every
        // scan gets an explicit pool or the Objective-C objects would leak.
        @autoreleasepool {
            CWInterface* iface = [[CWWiFiClient sharedWiFiClient] interface];
            if (iface == nil) {
                if (!warned_no_iface_) {
                    std::cerr << "[rf] No Wi-Fi interface available (is Wi-Fi turned on?)\n";
                    warned_no_iface_ = true;
                }
                return false;
            }

            NSError* err = nil;
            NSSet<CWNetwork*>* networks = [iface scanForNetworksWithName:nil error:&err];
            if (networks == nil) {
                std::cerr << "[rf] Wi-Fi scan failed: "
                          << (err ? err.localizedDescription.UTF8String : "unknown error") << "\n";
                return false;
            }

            std::vector<ApReading> aps;
            aps.reserve(networks.count);
            size_t no_bssid = 0, unkeyed = 0;

            for (CWNetwork* net in networks) {
                const NSInteger rssi = net.rssiValue;
                if (rssi >= 0) continue;  // 0 means "no valid reading"

                // Stable identity -> hash. Preferred: the BSSID (MAC address).
                uint64_t h = 0;
                NSString* bssid = net.bssid;
                if (bssid != nil) {
                    const char* s = bssid.lowercaseString.UTF8String;
                    h = fnv1a(s, std::strlen(s));
                } else {
                    // BSSID is nil when macOS withholds it (Location Services, see below).
                    ++no_bssid;
                    NSString* ssid = net.ssid;
                    NSData* ie = net.informationElementData;
                    if (ssid != nil) {
                        const char* s = ssid.UTF8String;
                        h = fnv1a(s, std::strlen(s));
                    } else if (ie != nil && ie.length > 0) {
                        // Best-effort: APs of the same make/model may share a cell.
                        h = fnv1a(ie.bytes, ie.length);
                    } else {
                        ++unkeyed;
                        continue;
                    }
                }

                ApReading ap;
                ap.x = static_cast<int>(h % kGridCols);
                ap.y = static_cast<int>((h >> 32) % kGridRows);
                ap.rssi = static_cast<float>(rssi);
                aps.push_back(ap);
            }

            if (no_bssid > 0 && !warned_redacted_) {
                warned_redacted_ = true;
                std::cerr << "[rf] " << no_bssid << " of " << networks.count
                          << " networks had no BSSID (" << unkeyed << " had no usable ID at all).\n"
                          << "[rf] macOS 14+ hides BSSID/SSID unless the process is authorized for "
                             "Location Services; falling back to SSID/IE hashing where possible.\n";
            }
            if (!first_scan_logged_) {
                first_scan_logged_ = true;
                std::cerr << "[rf] first scan: " << networks.count << " networks, "
                          << aps.size() << " placed on the grid\n";
            }

            out = render_blobs(aps);
            return true;
        }
    }

    void run() {
        while (true) {
            RfSnapshot next;
            const bool ok = scan_once(next);  // blocks 2-3 s; NO lock held, video thread unaffected

            std::unique_lock<std::mutex> lk(mtx_);
            if (stop_) break;
            if (ok) grid_ = next;  // brief critical section: copy 192 floats
            // Short pause after a good scan, longer after a failure; wakes early on shutdown.
            const auto pause = ok ? std::chrono::milliseconds(500) : std::chrono::milliseconds(3000);
            if (cv_.wait_for(lk, pause, [this] { return stop_; })) break;
        }
    }

    mutable std::mutex      mtx_;
    std::condition_variable cv_;
    RfSnapshot              grid_{};
    bool                    stop_ = false;

    // Only touched by the worker thread (no locking needed).
    bool warned_no_iface_   = false;
    bool warned_redacted_   = false;
    bool first_scan_logged_ = false;

    std::thread             worker_;  // declared last: constructed after everything it uses
};

// Draws the RF grid as a semi-transparent thermal heatmap over `frame` (in place).
//
// `smoothed` is persistent state owned by the caller: the grid now only changes
// every ~3 s (one Wi-Fi scan) while we render ~30 FPS, so we ease toward each
// new snapshot every frame instead of letting the heatmap jump.
static void overlay_thermal(cv::Mat& frame, const RfSnapshot& snap, cv::Mat& smoothed) {
    // 1) Snapshot -> 12x16 float Mat
    cv::Mat target(kGridRows, kGridCols, CV_32F);
    for (int r = 0; r < kGridRows; ++r)
        for (int c = 0; c < kGridCols; ++c)
            target.at<float>(r, c) = snap[r][c];

    // 2) Temporal smoothing: smoothed = 0.85 * smoothed + 0.15 * target
    if (smoothed.empty()) target.copyTo(smoothed);
    else cv::addWeighted(smoothed, 0.85, target, 0.15, 0.0, smoothed);

    // 3) Upscale 16x12 -> frame size with bicubic interpolation
    cv::Mat big;
    cv::resize(smoothed, big, frame.size(), 0, 0, cv::INTER_CUBIC);

    // 4) Normalize dBm [kRssiMin, kRssiMax] -> [0, 255] (8-bit).
//    A FIXED range is used on purpose to prevent noise amplification.
const double scale = 255.0 / (kRssiMax - kRssiMin);
cv::Mat u8;
big.convertTo(u8, CV_8U, scale, -kRssiMin * scale);

    // 5) Color map: blue (weak) -> red (strong)
    cv::Mat heat;
    cv::applyColorMap(u8, heat, cv::COLORMAP_JET);

    // 6) Alpha blend in place: frame = 0.70 * frame + 0.30 * heat
    cv::addWeighted(frame, 1.0 - kThermalAlpha, heat, kThermalAlpha, 0.0, frame);
}

// Finds the strongest cell and turns it into a sentence for the LLM prompt.
// Camera frames here are not mirrored, so "left" means the left of the image.
static std::string rf_prompt_hint(const RfSnapshot& s) {
    float peak = kRssiMin;
    int peak_col = 0;
    for (int r = 0; r < kGridRows; ++r)
        for (int c = 0; c < kGridCols; ++c)
            if (s[r][c] > peak) { peak = s[r][c]; peak_col = c; }

    if (peak < kStrongRssi) return "No strong RF signal is detected nearby.";

    std::string where;
    if (peak_col < kGridCols / 3)            where = "on the left";
    else if (peak_col >= 2 * kGridCols / 3)  where = "on the right";
    else                                     where = "in the center";

    return "A strong RF signal is detected " + where + " (about " +
           std::to_string(static_cast<int>(std::lround(peak))) + " dBm).";
}

// =============================================================================
// YOLO pre/post-processing (unchanged)
// =============================================================================

static cv::Mat letterbox(const cv::Mat& src, int model_w, int model_h, LetterboxInfo& info) {
    info.scale = std::min(static_cast<float>(model_w) / src.cols, static_cast<float>(model_h) / src.rows);
    const int new_w = static_cast<int>(std::round(src.cols * info.scale));
    const int new_h = static_cast<int>(std::round(src.rows * info.scale));
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    info.pad_x = (model_w - new_w) / 2;
    info.pad_y = (model_h - new_h) / 2;
    cv::Mat out;
    cv::copyMakeBorder(resized, out, info.pad_y, model_h - new_h - info.pad_y,
                       info.pad_x, model_w - new_w - info.pad_x, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return out;
}

static void preprocess(const cv::Mat& letterboxed, std::vector<float>& dst) {
    const int H = letterboxed.rows, W = letterboxed.cols;
    const size_t plane = static_cast<size_t>(H) * W;
    cv::Mat rgb, f32;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);
    float* base = dst.data();
    std::vector<cv::Mat> planes = {
        cv::Mat(H, W, CV_32FC1, base + 0 * plane),
        cv::Mat(H, W, CV_32FC1, base + 1 * plane),
        cv::Mat(H, W, CV_32FC1, base + 2 * plane)
    };
    cv::split(f32, planes);
}

static std::vector<Detection> postprocess(const float* output, int64_t num_channels, int64_t num_anchors,
                                          const LetterboxInfo& lb, int img_w, int img_h, float conf_threshold, float nms_threshold) {
    const int64_t N = num_anchors;
    const int num_classes = static_cast<int>(num_channels) - 4;
    const float* cx_row = output + 0 * N, * cy_row = output + 1 * N;
    const float* w_row = output + 2 * N, * h_row = output + 3 * N;
    const float* cls_row = output + 4 * N;
    std::vector<cv::Rect> boxes; std::vector<float> scores; std::vector<int> class_ids;
    boxes.reserve(256); scores.reserve(256); class_ids.reserve(256);
    for (int64_t i = 0; i < N; ++i) {
        const float* p = cls_row + i;
        int best_class = 0; float best_score = *p;
        for (int k = 1; k < num_classes; ++k) {
            p += N;
            if (*p > best_score) { best_score = *p; best_class = k; }
        }
        if (best_score < conf_threshold) continue;
        const float cx = cx_row[i], cy = cy_row[i], w = w_row[i], h = h_row[i];
        boxes.emplace_back(static_cast<int>(cx - w * 0.5f), static_cast<int>(cy - h * 0.5f),
                           static_cast<int>(w), static_cast<int>(h));
        scores.push_back(best_score); class_ids.push_back(best_class);
    }
    std::vector<int> kept;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, nms_threshold, kept);
    std::vector<Detection> detections;
    detections.reserve(kept.size());
    for (int idx : kept) {
        const cv::Rect& b = boxes[idx];
        float x1 = std::clamp((b.x - lb.pad_x) / lb.scale, 0.0f, static_cast<float>(img_w));
        float y1 = std::clamp((b.y - lb.pad_y) / lb.scale, 0.0f, static_cast<float>(img_h));
        float x2 = std::clamp((b.x + b.width - lb.pad_x) / lb.scale, 0.0f, static_cast<float>(img_w));
        float y2 = std::clamp((b.y + b.height - lb.pad_y) / lb.scale, 0.0f, static_cast<float>(img_h));
        detections.push_back(Detection{x1, y1, x2 - x1, y2 - y1, class_ids[idx], scores[idx]});
    }
    return detections;
}

// =============================================================================
// llama.cpp generation (unchanged)
// =============================================================================

static void generate_llm_response(llama_context* ctx, const std::string& prompt) {
    const llama_model* model = llama_get_model(ctx);
    const llama_vocab* vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens_list(prompt.size() + 3);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens_list.data(), tokens_list.size(), true, true);
    if (n_tokens < 0) {
        tokens_list.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens_list.data(), tokens_list.size(), true, true);
    }
    tokens_list.resize(n_tokens);
    llama_batch batch = llama_batch_init(tokens_list.size(), 0, 1);
    batch.n_tokens = tokens_list.size();
    for (size_t i = 0; i < tokens_list.size(); i++) {
        batch.token[i] = tokens_list[i]; batch.pos[i] = i;
        batch.n_seq_id[i]= 1; batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == tokens_list.size() - 1);
    }
    if (llama_decode(ctx, batch) != 0) { std::cerr << "[LLM Error] llama_decode failed\n"; llama_batch_free(batch); return; }
    std::cout << "\n[LLM Scene Description]: ";
    int n_cur = batch.n_tokens, n_decode = 0;
    while (n_cur < 256 && n_decode < 64) {
        auto* logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        int n_vocab = llama_vocab_n_tokens(vocab);
        llama_token new_token_id = 0; float max_logit = logits[0];
        for (int i = 1; i < n_vocab; ++i) { if (logits[i] > max_logit) { max_logit = logits[i]; new_token_id = i; } }
        if (llama_vocab_is_eog(vocab, new_token_id)) break;
        char buf[128]; int nChars = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (nChars > 0) std::cout << std::string(buf, nChars) << std::flush;
        batch.n_tokens = 1; batch.token[0] = new_token_id; batch.pos[0] = n_cur;
        batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = true;
        if (llama_decode(ctx, batch) != 0) break;
        n_cur++; n_decode++;
    }
    std::cout << "\n----------------------------------------\n";
    llama_batch_free(batch);
}

// =============================================================================
// main (unchanged from the Phase 1 merge)
// =============================================================================

int main(int argc, char** argv) {
    const std::string model_path = "models/yolov8n.onnx";
    const std::string llm_path = "models/qwen1_5-1_8b-chat-q4_0.gguf";
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    std::cout << "[info] Loading LLM from " << llm_path << "...\n";
    llama_model* model = llama_model_load_from_file(llm_path.c_str(), model_params);
    if (!model) return 1;
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512; ctx_params.n_batch = 512;
    llama_context* l_ctx = llama_init_from_model(model, ctx_params);
    if (!l_ctx) { llama_model_free(model); return 1; }
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolov8n_cam");
        Ort::SessionOptions so; so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        std::unordered_map<std::string, std::string> coreml_opts = { {"ModelFormat", "MLProgram"}, {"MLComputeUnits", "ALL"}, {"RequireStaticInputShapes", "1"} };
        so.AppendExecutionProvider("CoreML", coreml_opts);
        Ort::Session session(env, model_path.c_str(), so);
        Ort::AllocatorWithDefaultOptions allocator;
        auto in_name_ptr = session.GetInputNameAllocated(0, allocator);
        auto out_name_ptr = session.GetOutputNameAllocated(0, allocator);
        const char* input_names[] = {in_name_ptr.get()};
        const char* output_names[] = {out_name_ptr.get()};
        auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        const int model_h = static_cast<int>(input_shape[2]), model_w = static_cast<int>(input_shape[3]);
        std::vector<float> input_buffer(static_cast<size_t>(3) * model_h * model_w);
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, input_buffer.data(), input_buffer.size(), input_shape.data(), input_shape.size());
        cv::VideoCapture cap(0, cv::CAP_AVFOUNDATION);
        if (!cap.isOpened()) return 1;

        // RF scanner: starts its background thread here; joined automatically
        // when it goes out of scope (end of this try block or any early exit).
        RfSimulator rf;
        cv::Mat rf_smoothed;  // persistent state for temporal smoothing of the heatmap

        cv::Mat frame; int frame_count = 0;
        while (true) {
            cap >> frame; if (frame.empty()) break;
            LetterboxInfo lb{};
            cv::Mat boxed = letterbox(frame, model_w, model_h, lb);
            preprocess(boxed, input_buffer);
            auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
            const auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            const float* out_data = outputs[0].GetTensorData<float>();
            std::vector<Detection> dets = postprocess(out_data, out_shape[1], out_shape[2], lb, frame.cols, frame.rows, 0.25f, 0.45f);

            // RF Ghost Vision: one locked snapshot per frame, used by the overlay and the LLM hint.
            // Runs after inference (YOLO saw the clean frame) and before the boxes are drawn.
            const RfSnapshot rf_snap = rf.snapshot();
            overlay_thermal(frame, rf_snap, rf_smoothed);

            for (const Detection& d : dets) {
                cv::Rect r(static_cast<int>(d.x), static_cast<int>(d.y), static_cast<int>(d.width), static_cast<int>(d.height));
                cv::rectangle(frame, r, cv::Scalar(0, 255, 0), 2);
                char label[64]; std::snprintf(label, sizeof(label), "%s %.2f", kClassNames[d.class_id], d.confidence);
                cv::putText(frame, label, cv::Point(r.x, std::max(r.y - 6, 12)), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2);
            }
            cv::imshow("YOLOv8n + Qwen LLM | CoreML + Metal", frame);
            frame_count++;
            if (frame_count >= 75 && !dets.empty()) {
                frame_count = 0;
                std::ostringstream prompt_ss;
                prompt_ss << "A computer vision camera detected the following objects: ";
                for (size_t i = 0; i < dets.size(); ++i) { prompt_ss << kClassNames[dets[i].class_id]; if (i + 1 < dets.size()) prompt_ss << ", "; }
                prompt_ss << ". " << rf_prompt_hint(rf_snap);
                prompt_ss << " Give a brief, one-sentence description of what is happening.";
                generate_llm_response(l_ctx, prompt_ss.str());
            }
            if (cv::waitKey(1) == 27) break;
        }
        llama_free(l_ctx); llama_model_free(model); llama_backend_free();
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << "\n"; return 1; }
    return 0;
}
