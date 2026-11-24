#include "<tree>/src/Sharplight.Shaders/Kernels/gemm_fp4_native.cu"
#include <cstdio>
void launch_g(const unsigned char* x, const unsigned char* w, float* y, unsigned T, cudaStream_t s){
  F4nParams p; p.x=x; p.w=w; p.y=y; p.T=T;
#if DUAL
  p.w2=w+(size_t)F4_WRB*N;
#endif
  size_t smem = F4_STAGES*F4_STAGE_BYTES;
  static bool init=false; if(!init){ cudaFuncSetAttribute(gemm_fp4_native, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); init=true;}
  dim3 grid((T+F4_BM-1)/F4_BM, (N+F4_BOUT-1)/F4_BOUT);
  gemm_fp4_native<<<grid, F4_THREADS, smem, s>>>(p);
}
