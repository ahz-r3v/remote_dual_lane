#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            throw std::runtime_error(std::string(#expr) + " failed: " +         \
                                     cudaGetErrorString(_err));                 \
        }                                                                       \
    } while (0)

__global__ void reset_kernel(int *short_done,
                             int *full_counts,
                             int *cap_counts,
                             int *observed_done,
                             int chunks) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i == 0) {
        *short_done = 0;
    }
    for (int c = i; c < chunks; c += blockDim.x * gridDim.x) {
        full_counts[c] = 0;
        cap_counts[c] = 0;
        observed_done[c] = -1;
    }
}

__global__ void mark_short_done_kernel(int *short_done) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicExch(short_done, 1);
    }
}

__global__ void check_done_kernel(int *short_done,
                                  cudaGraphConditionalHandle handle,
                                  int *observed_done,
                                  int chunk) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int done = atomicAdd(short_done, 0);
        observed_done[chunk] = done;
        cudaGraphSetConditional(handle, done != 0);
    }
}

__global__ void branch_marker_kernel(int chunk,
                                     int branch,
                                     int *full_counts,
                                     int *cap_counts) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (branch != 0) {
            atomicAdd(full_counts + chunk, 1);
        } else {
            atomicAdd(cap_counts + chunk, 1);
        }
    }
}

static cudaGraphNode_t add_kernel_node(cudaGraph_t graph,
                                       const std::vector<cudaGraphNode_t> &deps,
                                       void *func,
                                       dim3 grid,
                                       dim3 block,
                                       void **args) {
    cudaGraphNodeParams params{};
    params.type = cudaGraphNodeTypeKernel;
    params.kernel.func = func;
    params.kernel.gridDim = grid;
    params.kernel.blockDim = block;
    params.kernel.sharedMemBytes = 0;
    params.kernel.kernelParams = args;
    params.kernel.extra = nullptr;
    cudaGraphNode_t node{};
    CUDA_CHECK(cudaGraphAddNode(&node, graph, deps.data(), deps.size(), &params));
    return node;
}

static cudaGraphNode_t add_event_record_node(cudaGraph_t graph,
                                             const std::vector<cudaGraphNode_t> &deps,
                                             cudaEvent_t event) {
    cudaGraphNodeParams params{};
    params.type = cudaGraphNodeTypeEventRecord;
    params.eventRecord.event = event;
    cudaGraphNode_t node{};
    CUDA_CHECK(cudaGraphAddNode(&node, graph, deps.data(), deps.size(), &params));
    return node;
}

static cudaGraphNode_t add_child_graph_node(
    cudaGraph_t graph,
    const std::vector<cudaGraphNode_t> &deps,
    cudaGraph_t child_graph) {
    cudaGraphNode_t node{};
    CUDA_CHECK(cudaGraphAddChildGraphNode(
        &node, graph, deps.data(), deps.size(), child_graph));
    return node;
}

struct KernelArgStorage {
    std::vector<int> chunk;
    std::vector<int> branch;
    std::vector<int *> short_done;
    std::vector<int *> full_counts;
    std::vector<int *> cap_counts;
    std::vector<int *> observed_done;
    std::vector<cudaGraphConditionalHandle> handles;
    std::vector<std::vector<void *>> argv;

    explicit KernelArgStorage(int reserve) {
        chunk.reserve(reserve);
        branch.reserve(reserve);
        short_done.reserve(reserve);
        full_counts.reserve(reserve);
        cap_counts.reserve(reserve);
        observed_done.reserve(reserve);
        handles.reserve(reserve);
        argv.reserve(reserve);
    }

    void **push_reset(int *done,
                      int *full,
                      int *cap,
                      int *observed,
                      int chunk_count) {
        short_done.push_back(done);
        full_counts.push_back(full);
        cap_counts.push_back(cap);
        observed_done.push_back(observed);
        chunk.push_back(chunk_count);
        argv.push_back({&short_done.back(), &full_counts.back(), &cap_counts.back(),
                        &observed_done.back(), &chunk.back()});
        return argv.back().data();
    }

    void **push_mark(int *done) {
        short_done.push_back(done);
        argv.push_back({&short_done.back()});
        return argv.back().data();
    }

    void **push_check(int *done,
                      cudaGraphConditionalHandle handle,
                      int *observed,
                      int c) {
        short_done.push_back(done);
        handles.push_back(handle);
        observed_done.push_back(observed);
        chunk.push_back(c);
        argv.push_back({&short_done.back(), &handles.back(), &observed_done.back(),
                        &chunk.back()});
        return argv.back().data();
    }

    void **push_branch_marker(int c,
                              int branch_id,
                              int *full,
                              int *cap) {
        chunk.push_back(c);
        branch.push_back(branch_id);
        full_counts.push_back(full);
        cap_counts.push_back(cap);
        argv.push_back({&chunk.back(), &branch.back(),
                        &full_counts.back(), &cap_counts.back()});
        return argv.back().data();
    }
};

class ConditionalCudaGraphBuilder {
  public:
    explicit ConditionalCudaGraphBuilder(int chunks) : chunks_(chunks), args_(6 * chunks + 16) {
        if (chunks_ <= 0) {
            throw std::runtime_error("chunks must be positive");
        }
        CUDA_CHECK(cudaMalloc(&short_done_, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&full_counts_, sizeof(int) * chunks_));
        CUDA_CHECK(cudaMalloc(&cap_counts_, sizeof(int) * chunks_));
        CUDA_CHECK(cudaMalloc(&observed_done_, sizeof(int) * chunks_));
        CUDA_CHECK(cudaStreamCreateWithFlags(&capture_stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&short_start_event_));
        CUDA_CHECK(cudaEventCreate(&short_end_event_));
        CUDA_CHECK(cudaEventCreate(&long_start_event_));
        CUDA_CHECK(cudaEventCreate(&long_end_event_));
        CUDA_CHECK(cudaGraphCreate(&graph_, 0));

        reset_node_ = add_kernel_node(
            graph_, {}, reinterpret_cast<void *>(reset_kernel), dim3(1), dim3(256),
            args_.push_reset(short_done_, full_counts_, cap_counts_, observed_done_, chunks_));
        short_start_node_ = add_event_record_node(graph_, {reset_node_}, short_start_event_);
        long_start_node_ = add_event_record_node(graph_, {reset_node_}, long_start_event_);
        long_prev_ = long_start_node_;
    }

    ~ConditionalCudaGraphBuilder() {
        if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
        if (graph_) cudaGraphDestroy(graph_);
        if (capture_stream_) cudaStreamDestroy(capture_stream_);
        if (short_start_event_) cudaEventDestroy(short_start_event_);
        if (short_end_event_) cudaEventDestroy(short_end_event_);
        if (long_start_event_) cudaEventDestroy(long_start_event_);
        if (long_end_event_) cudaEventDestroy(long_end_event_);
        if (short_done_) cudaFree(short_done_);
        if (full_counts_) cudaFree(full_counts_);
        if (cap_counts_) cudaFree(cap_counts_);
        if (observed_done_) cudaFree(observed_done_);
    }

    uintptr_t stream_ptr() const {
        return reinterpret_cast<uintptr_t>(capture_stream_);
    }

    void begin_short_capture() {
        ensure_not_instantiated();
        ensure_not_capturing();
        capture_graph_ = graph_;
        capture_before_ = graph_nodes(graph_);
        cudaGraphNode_t dep = short_start_node_;
        CUDA_CHECK(cudaStreamBeginCaptureToGraph(
            capture_stream_, graph_, &dep, nullptr, 1, cudaStreamCaptureModeRelaxed));
        capturing_ = true;
        capture_kind_ = "short";
    }

    void end_short_capture() {
        ensure_capturing("short");
        mark_short_done_kernel<<<dim3(1), dim3(1), 0, capture_stream_>>>(short_done_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(short_end_event_, capture_stream_));
        cudaGraph_t captured{};
        CUDA_CHECK(cudaStreamEndCapture(capture_stream_, &captured));
        capturing_ = false;
        short_mark_recorded_ = true;
        capture_kind_.clear();
    }

    void add_short_child_graph(uintptr_t raw_graph) {
        ensure_not_instantiated();
        ensure_not_capturing();
        if (short_mark_recorded_) {
            throw std::runtime_error("short capture was already added");
        }
        auto child_graph = reinterpret_cast<cudaGraph_t>(raw_graph);
        cudaGraphNode_t child = add_child_graph_node(
            graph_, {short_start_node_}, child_graph);
        cudaGraphNode_t mark = add_kernel_node(
            graph_, {child}, reinterpret_cast<void *>(mark_short_done_kernel),
            dim3(1), dim3(1), args_.push_mark(short_done_));
        short_end_node_ = add_event_record_node(graph_, {mark}, short_end_event_);
        short_mark_recorded_ = true;
    }

    void add_long_chunk(int chunk) {
        ensure_not_instantiated();
        if (chunk < 0 || chunk >= chunks_) {
            throw std::runtime_error("chunk out of range");
        }
        if (static_cast<int>(chunks_data_.size()) != chunk) {
            throw std::runtime_error("long chunks must be added in order");
        }

        cudaGraphConditionalHandle handle{};
        CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, graph_, 0,
                                                    cudaGraphCondAssignDefault));
        cudaGraphNode_t check = add_kernel_node(
            graph_, {long_prev_}, reinterpret_cast<void *>(check_done_kernel),
            dim3(1), dim3(1), args_.push_check(short_done_, handle, observed_done_, chunk));

        cudaGraphNodeParams cond_params{};
        cond_params.type = cudaGraphNodeTypeConditional;
        cond_params.conditional.handle = handle;
        cond_params.conditional.type = cudaGraphCondTypeIf;
        cond_params.conditional.size = 2;
        cudaGraphNode_t cond{};
        CUDA_CHECK(cudaGraphAddNode(&cond, graph_, &check, 1, &cond_params));

        ChunkData data{};
        data.cond = cond;
        data.full_body = cond_params.conditional.phGraph_out[0];
        data.cap_body = cond_params.conditional.phGraph_out[1];
        chunks_data_.push_back(data);
        long_prev_ = cond;
    }

    void begin_long_branch_capture(int chunk, bool full_branch) {
        ensure_not_instantiated();
        ensure_not_capturing();
        if (chunk < 0 || chunk >= static_cast<int>(chunks_data_.size())) {
            throw std::runtime_error("chunk has not been added");
        }
        capture_graph_ = full_branch ? chunks_data_[chunk].full_body
                                     : chunks_data_[chunk].cap_body;
        capture_before_ = graph_nodes(capture_graph_);
        capture_chunk_ = chunk;
        capture_full_branch_ = full_branch;
        CUDA_CHECK(cudaStreamBeginCaptureToGraph(
            capture_stream_, capture_graph_, nullptr, nullptr, 0,
            cudaStreamCaptureModeRelaxed));
        capturing_ = true;
        capture_kind_ = full_branch ? "long_full" : "long_cap";
    }

    void end_long_branch_capture() {
        ensure_capturing_any_long();
        cudaGraph_t captured{};
        CUDA_CHECK(cudaStreamEndCapture(capture_stream_, &captured));
        add_kernel_node(
            capture_graph_, {}, reinterpret_cast<void *>(branch_marker_kernel),
            dim3(1), dim3(1),
            args_.push_branch_marker(capture_chunk_, capture_full_branch_ ? 1 : 0,
                                     full_counts_, cap_counts_));
        capturing_ = false;
        capture_kind_.clear();
        capture_chunk_ = -1;
        capture_full_branch_ = false;
    }

    void add_long_branch_child_graph(int chunk, bool full_branch, uintptr_t raw_graph) {
        ensure_not_instantiated();
        ensure_not_capturing();
        if (chunk < 0 || chunk >= static_cast<int>(chunks_data_.size())) {
            throw std::runtime_error("chunk has not been added");
        }
        auto body_graph = full_branch ? chunks_data_[chunk].full_body
                                      : chunks_data_[chunk].cap_body;
        auto child_graph = reinterpret_cast<cudaGraph_t>(raw_graph);
        cudaGraphNode_t child = add_child_graph_node(body_graph, {}, child_graph);
        add_kernel_node(
            body_graph, {child}, reinterpret_cast<void *>(branch_marker_kernel),
            dim3(1), dim3(1),
            args_.push_branch_marker(chunk, full_branch ? 1 : 0,
                                     full_counts_, cap_counts_));
    }

    void instantiate() {
        ensure_not_capturing();
        if (static_cast<int>(chunks_data_.size()) != chunks_) {
            throw std::runtime_error("not all long chunks were added");
        }
        if (!short_mark_recorded_) {
            throw std::runtime_error("short capture was not added");
        }
        if (!long_end_node_) {
            long_end_node_ = add_event_record_node(graph_, {long_prev_}, long_end_event_);
        }
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, 0));
    }

    py::dict replay(int repeats, int warmup, int launch_priority) {
        if (!graph_exec_) {
            instantiate();
        }
        repeats = repeats > 0 ? repeats : 1;
        warmup = warmup >= 0 ? warmup : 0;

        cudaStream_t launch_stream{};
        CUDA_CHECK(cudaStreamCreateWithPriority(&launch_stream, cudaStreamNonBlocking,
                                                launch_priority));
        cudaEvent_t start{}, stop{};
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        for (int i = 0; i < warmup; ++i) {
            CUDA_CHECK(cudaGraphLaunch(graph_exec_, launch_stream));
            CUDA_CHECK(cudaStreamSynchronize(launch_stream));
        }

        float total_ms = 0.0f;
        for (int i = 0; i < repeats; ++i) {
            CUDA_CHECK(cudaEventRecord(start, launch_stream));
            CUDA_CHECK(cudaGraphLaunch(graph_exec_, launch_stream));
            CUDA_CHECK(cudaEventRecord(stop, launch_stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            total_ms += ms;
        }
        CUDA_CHECK(cudaStreamSynchronize(launch_stream));

        std::vector<int> h_full(chunks_), h_cap(chunks_), h_observed(chunks_);
        CUDA_CHECK(cudaMemcpy(h_full.data(), full_counts_, sizeof(int) * chunks_,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_cap.data(), cap_counts_, sizeof(int) * chunks_,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_observed.data(), observed_done_, sizeof(int) * chunks_,
                              cudaMemcpyDeviceToHost));

        int cap_chunks = 0;
        int full_chunks = 0;
        std::vector<std::string> branch(chunks_);
        for (int c = 0; c < chunks_; ++c) {
            if (h_full[c] > 0) {
                branch[c] = "full";
                full_chunks += 1;
            } else if (h_cap[c] > 0) {
                branch[c] = "cap";
                cap_chunks += 1;
            } else {
                branch[c] = "unknown";
            }
        }

        float short_ms = 0.0f;
        float long_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&short_ms, short_start_event_, short_end_event_));
        CUDA_CHECK(cudaEventElapsedTime(&long_ms, long_start_event_, long_end_event_));

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaStreamDestroy(launch_stream));

        py::dict out;
        out["chunks"] = chunks_;
        out["repeats"] = repeats;
        out["warmup"] = warmup;
        out["mean_ms"] = total_ms / static_cast<float>(repeats);
        out["short_ms"] = short_ms;
        out["long_ms"] = long_ms;
        out["total_ms"] = total_ms / static_cast<float>(repeats);
        out["observed_done"] = h_observed;
        out["cap_counts"] = h_cap;
        out["full_counts"] = h_full;
        out["branch"] = branch;
        out["cap_chunks"] = cap_chunks;
        out["full_chunks"] = full_chunks;
        return out;
    }

  private:
    struct ChunkData {
        cudaGraphNode_t cond{};
        cudaGraph_t full_body{};
        cudaGraph_t cap_body{};
    };

    std::unordered_set<cudaGraphNode_t> graph_nodes(cudaGraph_t graph) {
        size_t n = 0;
        CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &n));
        std::vector<cudaGraphNode_t> nodes(n);
        if (n > 0) {
            CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &n));
        }
        return std::unordered_set<cudaGraphNode_t>(nodes.begin(), nodes.end());
    }

    std::vector<cudaGraphNode_t> new_terminal_nodes(
        cudaGraph_t graph, const std::unordered_set<cudaGraphNode_t> &before) {
        auto after = graph_nodes(graph);
        std::vector<cudaGraphNode_t> terminals;
        for (auto node : after) {
            if (before.find(node) != before.end()) {
                continue;
            }
            size_t dep_count = 0;
            CUDA_CHECK(cudaGraphNodeGetDependentNodes(node, nullptr, &dep_count));
            std::vector<cudaGraphNode_t> deps(dep_count);
            if (dep_count > 0) {
                CUDA_CHECK(cudaGraphNodeGetDependentNodes(node, deps.data(), &dep_count));
            }
            bool has_new_dependent = false;
            for (auto dep : deps) {
                if (before.find(dep) == before.end()) {
                    has_new_dependent = true;
                    break;
                }
            }
            if (!has_new_dependent) {
                terminals.push_back(node);
            }
        }
        return terminals;
    }

    void ensure_not_capturing() {
        if (capturing_) {
            throw std::runtime_error("a capture is already active");
        }
    }

    void ensure_capturing(const std::string &kind) {
        if (!capturing_ || capture_kind_ != kind) {
            throw std::runtime_error("expected active " + kind + " capture");
        }
    }

    void ensure_capturing_any_long() {
        if (!capturing_ ||
            (capture_kind_ != "long_cap" && capture_kind_ != "long_full")) {
            throw std::runtime_error("expected active long branch capture");
        }
    }

    void ensure_not_instantiated() {
        if (graph_exec_) {
            throw std::runtime_error("graph has already been instantiated");
        }
    }

    int chunks_{0};
    KernelArgStorage args_;
    cudaGraph_t graph_{};
    cudaGraphExec_t graph_exec_{};
    cudaStream_t capture_stream_{};
    cudaGraphNode_t reset_node_{};
    cudaGraphNode_t short_start_node_{};
    cudaGraphNode_t short_end_node_{};
    cudaGraphNode_t long_start_node_{};
    cudaGraphNode_t long_end_node_{};
    cudaGraphNode_t short_mark_node_{};
    cudaGraphNode_t long_prev_{};
    cudaEvent_t short_start_event_{};
    cudaEvent_t short_end_event_{};
    cudaEvent_t long_start_event_{};
    cudaEvent_t long_end_event_{};
    int *short_done_{};
    int *full_counts_{};
    int *cap_counts_{};
    int *observed_done_{};
    std::vector<ChunkData> chunks_data_;
    bool capturing_{false};
    bool short_mark_recorded_{false};
    std::string capture_kind_;
    cudaGraph_t capture_graph_{};
    std::unordered_set<cudaGraphNode_t> capture_before_;
    int capture_chunk_{-1};
    bool capture_full_branch_{false};
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<ConditionalCudaGraphBuilder>(m, "ConditionalCudaGraphBuilder")
        .def(py::init<int>(), py::arg("chunks"))
        .def("stream_ptr", &ConditionalCudaGraphBuilder::stream_ptr)
        .def("begin_short_capture", &ConditionalCudaGraphBuilder::begin_short_capture)
        .def("end_short_capture", &ConditionalCudaGraphBuilder::end_short_capture)
        .def("add_short_child_graph",
             &ConditionalCudaGraphBuilder::add_short_child_graph,
             py::arg("raw_graph"))
        .def("add_long_chunk", &ConditionalCudaGraphBuilder::add_long_chunk)
        .def("begin_long_branch_capture",
             &ConditionalCudaGraphBuilder::begin_long_branch_capture,
             py::arg("chunk"), py::arg("full_branch"))
        .def("end_long_branch_capture",
             &ConditionalCudaGraphBuilder::end_long_branch_capture)
        .def("add_long_branch_child_graph",
             &ConditionalCudaGraphBuilder::add_long_branch_child_graph,
             py::arg("chunk"), py::arg("full_branch"), py::arg("raw_graph"))
        .def("instantiate", &ConditionalCudaGraphBuilder::instantiate)
        .def("replay", &ConditionalCudaGraphBuilder::replay,
             py::arg("repeats") = 1, py::arg("warmup") = 0,
             py::arg("launch_priority") = 0);
}
