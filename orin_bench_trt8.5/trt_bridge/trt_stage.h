// C ABI over the TensorRT 8.5 C++ runtime, so Python 3.12 (PyTorch) can drive engines in its own process.
// JetPack 5.1 ships TensorRT Python bindings for Python 3.8 only; this bridge avoids needing them.
// All buffers are caller-owned CUDA pointers: nothing is copied. Functions returning int give 0 on success;
// the message for the last failure is in trts_last_error().
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TrtStage TrtStage;

const char* trts_last_error(void);
TrtStage* trts_load(const char* engine_path);  // NULL on failure
void trts_free(TrtStage* s);

int trts_n_io(const TrtStage* s);
const char* trts_io_name(const TrtStage* s, int i);
int trts_io_is_input(const TrtStage* s, int i);
// TensorRT 8.5 DataType codes: FLOAT=0, HALF=1, INT8=2, INT32=3, BOOL=4, UINT8=5.
int trts_io_dtype(const TrtStage* s, int i);
int trts_io_rank(const TrtStage* s, int i);
int64_t trts_io_dim(const TrtStage* s, int i, int d);

int trts_bind(TrtStage* s, const char* name, void* cuda_ptr);
// Records one pass into a CUDA graph. Every buffer must be bound, and must stay at the same address afterwards.
int trts_enable_graph(TrtStage* s);
int trts_run(TrtStage* s);  // launches and waits for completion

#ifdef __cplusplus
}
#endif
