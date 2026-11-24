
#include "<tree>/src/Sharplight.Shaders/Kernels/quantize_fp4.cu"
void launch_q(const float* x, unsigned char* q, unsigned T){ Q4nParams p{x,q,T}; quantize_fp4<<<T,128>>>(p); }
