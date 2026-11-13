// Generates Fixtures/ggml_quant.bin from llama.cpp (ggml_quantize_chunk + type traits to_float).
// gcc gen_quant.c -I<llama.cpp>/ggml/include -L<build>/bin -lggml-base -lggml-cpu -lggml -lm -o gen && ./gen ggml_quant.bin
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "ggml.h"
#include "ggml-cpu.h"
static uint32_t s=12345;
static float rnd(void){ s=s*1664525u+1013904223u; return ((s>>8)&0xFFFF)/65536.0f; }
static float gauss(void){ float u=rnd()+1e-6f, v=rnd(); return sqrtf(-2*logf(u))*cosf(6.2831853f*v); }
int main(int argc,char**argv){
  ggml_cpu_init();
  int types[] = {GGML_TYPE_Q4_0,GGML_TYPE_Q4_1,GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q8_0,GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K};
  FILE*f=fopen(argv[1],"wb");
  int n=sizeof(types)/sizeof(int);
  fwrite(&n,4,1,f);
  int rows=4, cols=512;
  for(int t=0;t<n;t++){
    enum ggml_type ty=types[t];
    float*src=malloc(rows*cols*4);
    for(int i=0;i<rows*cols;i++) src[i]=gauss()*(1+ (i/cols)) + 0.3f*(i%cols>256);
    size_t rb=ggml_row_size(ty,cols);
    void*q=malloc(rb*rows);
    ggml_quantize_chunk(ty,src,q,0,rows,cols,NULL);
    float*dq=malloc(rows*cols*4);
    const struct ggml_type_traits*tr=ggml_get_type_traits(ty);
    for(int r=0;r<rows;r++) tr->to_float((char*)q+r*rb,dq+r*cols,cols);
    int id=(int)ty; fwrite(&id,4,1,f); fwrite(&rows,4,1,f); fwrite(&cols,4,1,f);
    fwrite(src,4,rows*cols,f); fwrite(q,1,rb*rows,f); fwrite(dq,4,rows*cols,f);
  }
  fclose(f); return 0;
}
