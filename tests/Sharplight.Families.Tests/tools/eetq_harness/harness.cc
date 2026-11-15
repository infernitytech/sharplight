// Exposes EETQ's own symmetric_quantize (csrc/cutlass_kernels/cutlass_preprocessors.cc, Apache-2.0) for int8 on an sm80 layout.
#include "cuda_utils.h"
#include "cutlass_preprocessors.h"
#include <cstring>
using namespace fastertransformer;
extern "C" void eetq_quant_int8(const uint16_t* weight_f16, int rows, int cols, int8_t* processed, int8_t* unprocessed, uint16_t* scales) {
    symmetric_quantize<half, half>(processed, unprocessed, reinterpret_cast<half*>(scales), reinterpret_cast<const half*>(weight_f16),
                                   {(size_t)rows, (size_t)cols}, QuantType::INT8_WEIGHT_ONLY);
}
extern "C" void eetq_preprocess_int8(const int8_t* row_major, int rows, int cols, int8_t* processed) {
    preprocess_weights(processed, row_major, rows, cols, false, 80);
}
