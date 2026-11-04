#include "trt_stage.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {
thread_local std::string g_error;

int fail(const std::string& msg) {
    g_error = msg;
    return 1;
}

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) fprintf(stderr, "[trt] %s\n", msg);
    }
};
Logger g_logger;

// Serialized engines are several GB. Mapping the file avoids a second host copy, which matters on a
// unified-memory device where host and GPU share the same RAM.
struct MappedFile {
    int fd = -1;
    void* data = MAP_FAILED;
    size_t size = 0;
    MappedFile() = default;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    ~MappedFile() { release(); }
    void release() {
        if (data != MAP_FAILED) munmap(data, size);
        if (fd >= 0) close(fd);
        data = MAP_FAILED;
        fd = -1;
        size = 0;
    }
    bool open_file(const char* path) {
        fd = open(path, O_RDONLY);
        if (fd < 0) return !fail(std::string("cannot open ") + path + ": " + std::strerror(errno));
        struct stat info {};
        if (fstat(fd, &info) != 0) return !fail(std::string("cannot stat ") + path + ": " + std::strerror(errno));
        if (info.st_size <= 0 || !S_ISREG(info.st_mode))
            return !fail(std::string("engine is not a nonempty regular file: ") + path);
        size = static_cast<size_t>(info.st_size);
        data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) return !fail(std::string("cannot map ") + path + ": " + std::strerror(errno));
        return true;
    }
};
}  // namespace

struct TrtStage {
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    std::vector<std::string> names;
};

const char* trts_last_error(void) { return g_error.c_str(); }

void trts_free(TrtStage* s) {
    if (!s) return;
    if (s->stream) cudaStreamSynchronize(s->stream);
    if (s->graph_exec) cudaGraphExecDestroy(s->graph_exec);
    if (s->graph) cudaGraphDestroy(s->graph);
    if (s->context) s->context->destroy();
    if (s->engine) s->engine->destroy();
    if (s->runtime) s->runtime->destroy();
    if (s->stream) cudaStreamDestroy(s->stream);
    delete s;
}

TrtStage* trts_load(const char* engine_path) {
    if (!engine_path) {
        fail("trts_load requires an engine path");
        return nullptr;
    }
    TrtStage* s = nullptr;
    try {
        MappedFile file;
        if (!file.open_file(engine_path)) return nullptr;
        s = new TrtStage();
        cudaError_t err = cudaStreamCreateWithFlags(&s->stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fail(std::string("stream: ") + cudaGetErrorString(err));
            trts_free(s);
            return nullptr;
        }
        s->runtime = nvinfer1::createInferRuntime(g_logger);
        if (s->runtime) s->engine = s->runtime->deserializeCudaEngine(file.data, file.size);
        file.release();
        if (s->engine) s->context = s->engine->createExecutionContext();
        if (!s->context) {
            fail(std::string("could not create an execution context from ") + engine_path);
            trts_free(s);
            return nullptr;
        }
        for (int i = 0; i < s->engine->getNbIOTensors(); ++i) s->names.emplace_back(s->engine->getIOTensorName(i));
        return s;
    } catch (const std::exception& e) {
        fail(std::string("engine load: ") + e.what());
    } catch (...) {
        fail("engine load: unknown exception");
    }
    trts_free(s);
    return nullptr;
}

int trts_n_io(const TrtStage* s) { return static_cast<int>(s->names.size()); }
const char* trts_io_name(const TrtStage* s, int i) { return s->names[i].c_str(); }

int trts_io_is_input(const TrtStage* s, int i) {
    return s->engine->getTensorIOMode(s->names[i].c_str()) == nvinfer1::TensorIOMode::kINPUT;
}

int trts_io_dtype(const TrtStage* s, int i) {
    return static_cast<int>(s->engine->getTensorDataType(s->names[i].c_str()));
}

int trts_io_rank(const TrtStage* s, int i) { return s->engine->getTensorShape(s->names[i].c_str()).nbDims; }
int64_t trts_io_dim(const TrtStage* s, int i, int d) { return s->engine->getTensorShape(s->names[i].c_str()).d[d]; }

int trts_bind(TrtStage* s, const char* name, void* cuda_ptr) {
    if (s->graph_exec) return fail("buffers cannot change after trts_enable_graph");
    const bool is_input = s->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
    const bool ok = is_input ? s->context->setInputTensorAddress(name, cuda_ptr) : s->context->setTensorAddress(name, cuda_ptr);
    return ok ? 0 : fail(std::string("could not bind ") + name);
}

int trts_enable_graph(TrtStage* s) {
    if (s->graph_exec) return 0;
    // TensorRT initializes lazily on the first enqueue, which must not happen during capture.
    if (!s->context->enqueueV3(s->stream)) return fail("first enqueue failed");
    cudaError_t err = cudaStreamSynchronize(s->stream);
    if (err != cudaSuccess) return fail(std::string("sync: ") + cudaGetErrorString(err));
    err = cudaStreamBeginCapture(s->stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) return fail(std::string("begin capture: ") + cudaGetErrorString(err));
    const bool ok = s->context->enqueueV3(s->stream);
    err = cudaStreamEndCapture(s->stream, &s->graph);
    if (!ok || err != cudaSuccess) {
        if (s->graph) cudaGraphDestroy(s->graph);
        s->graph = nullptr;
        return fail(!ok ? "enqueue during capture failed" : std::string("end capture: ") + cudaGetErrorString(err));
    }
    err = cudaGraphInstantiate(&s->graph_exec, s->graph, 0);
    if (err != cudaSuccess) {
        cudaGraphDestroy(s->graph);
        s->graph = nullptr;
        return fail(std::string("instantiate: ") + cudaGetErrorString(err));
    }
    return 0;
}

int trts_run(TrtStage* s) {
    if (s->graph_exec) {
        cudaError_t err = cudaGraphLaunch(s->graph_exec, s->stream);
        if (err != cudaSuccess) return fail(std::string("launch: ") + cudaGetErrorString(err));
    } else if (!s->context->enqueueV3(s->stream)) {
        return fail("enqueue failed");
    }
    cudaError_t err = cudaStreamSynchronize(s->stream);
    return err == cudaSuccess ? 0 : fail(std::string("sync: ") + cudaGetErrorString(err));
}
