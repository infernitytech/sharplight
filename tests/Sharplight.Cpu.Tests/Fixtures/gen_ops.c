// Generates Fixtures/ggml_ops.bin (rope, gated_delta_net, ssm_conv outputs of the ggml CPU backend). Build as gen_quant.c.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "ggml.h"
#include "ggml-cpu.h"
static uint32_t s=777;
static float rnd(void){ s=s*1664525u+1013904223u; return ((s>>8)&0xFFFF)/65536.0f; }
static float sym(void){ return rnd()*2-1; }
static FILE*f;
static void wi(int v){ fwrite(&v,4,1,f); }
static void wf(const float*p,int n){ fwrite(p,4,n,f); }
static struct ggml_context* mk(void){ struct ggml_init_params ip={ .mem_size=256*1024*1024, .mem_buffer=NULL, .no_alloc=false}; return ggml_init(ip); }
static void run(struct ggml_context*c, struct ggml_tensor*out){ struct ggml_cgraph*g=ggml_new_graph(c); ggml_build_forward_expand(g,out); ggml_graph_compute_with_ctx(c,g,4); }

static void rope_case(int mode,int dims,int sec[4],float fscale,int T,int H,int D){
  struct ggml_context*c=mk();
  struct ggml_tensor*a=ggml_new_tensor_3d(c,GGML_TYPE_F32,D,H,T);
  for(int i=0;i<D*H*T;i++) ((float*)a->data)[i]=sym();
  struct ggml_tensor*p=ggml_new_tensor_1d(c,GGML_TYPE_I32,T*4);
  int32_t pos[4*64]; // [T,4] row-major copy for the file
  for(int t=0;t<T;t++) for(int ax=0;ax<4;ax++){ pos[t*4+ax]=(int)(rnd()*400); }
  for(int t=0;t<T;t++) for(int ax=0;ax<4;ax++) ((int32_t*)p->data)[t+T*ax]=pos[t*4+ax];
  struct ggml_tensor*o;
  int secs[4]={sec[0],sec[1],sec[2],sec[3]};
  if(mode==0||mode==2){ o=ggml_rope_ext(c,a,ggml_new_tensor_1d(c,GGML_TYPE_I32,T),NULL,dims,mode,0,10000.f,fscale,0,1,0,0);
     for(int t=0;t<T;t++) ((int32_t*)o->src[1]->data)[t]=pos[t*4]; }
  else o=ggml_rope_multi(c,a,p,NULL,dims,secs,mode,0,10000.f,fscale,0,1,0,0);
  run(c,o);
  wi(1); wi(mode); wi(dims); wi(sec[0]);wi(sec[1]);wi(sec[2]);wi(sec[3]); wf(&fscale,1); wi(T);wi(H);wi(D);
  wf((float*)a->data,D*H*T); fwrite(pos,4,T*4,f); wf((float*)o->data,D*H*T);
  ggml_free(c);
}
static void delta_case(int T,int Hk,int Hv,int S){
  struct ggml_context*c=mk();
  struct ggml_tensor*q=ggml_new_tensor_4d(c,GGML_TYPE_F32,S,Hk,T,1), *k=ggml_new_tensor_4d(c,GGML_TYPE_F32,S,Hk,T,1), *v=ggml_new_tensor_4d(c,GGML_TYPE_F32,S,Hv,T,1);
  struct ggml_tensor*g=ggml_new_tensor_4d(c,GGML_TYPE_F32,1,Hv,T,1), *b=ggml_new_tensor_4d(c,GGML_TYPE_F32,1,Hv,T,1), *st=ggml_new_tensor_4d(c,GGML_TYPE_F32,S,S,Hv,1);
  for(int i=0;i<S*Hk*T;i++){ ((float*)q->data)[i]=sym()*0.5f; ((float*)k->data)[i]=sym()*0.5f; }
  for(int i=0;i<S*Hv*T;i++) ((float*)v->data)[i]=sym();
  for(int i=0;i<Hv*T;i++){ ((float*)g->data)[i]=-rnd(); ((float*)b->data)[i]=rnd(); }
  for(int i=0;i<S*S*Hv;i++) ((float*)st->data)[i]=sym()*0.3f;
  struct ggml_tensor*o=ggml_gated_delta_net(c,q,k,v,g,b,st,1);
  run(c,o);
  wi(2); wi(T);wi(Hk);wi(Hv);wi(S);
  wf((float*)q->data,S*Hk*T); wf((float*)k->data,S*Hk*T); wf((float*)v->data,S*Hv*T); wf((float*)g->data,Hv*T); wf((float*)b->data,Hv*T); wf((float*)st->data,S*S*Hv);
  wf((float*)o->data,S*Hv*T + S*S*Hv);
  ggml_free(c);
}
static void conv_case(int T,int C,int K){
  struct ggml_context*c=mk();
  struct ggml_tensor*sx=ggml_new_tensor_3d(c,GGML_TYPE_F32,K-1+T,C,1), *w=ggml_new_tensor_2d(c,GGML_TYPE_F32,K,C);
  for(int i=0;i<(K-1+T)*C;i++) ((float*)sx->data)[i]=sym();
  for(int i=0;i<K*C;i++) ((float*)w->data)[i]=sym();
  struct ggml_tensor*o=ggml_ssm_conv(c,sx,w);
  run(c,o);
  wi(3); wi(T);wi(C);wi(K);
  // file layout: state [K-1,C] row-major, x [T,C], w [C,K], out [T,C]
  float*st=malloc(4*(K-1)*C),*x=malloc(4*T*C);
  for(int cc=0;cc<C;cc++){ for(int j=0;j<K-1;j++) st[j*C+cc]=((float*)sx->data)[cc*(K-1+T)+j]; for(int t=0;t<T;t++) x[t*C+cc]=((float*)sx->data)[cc*(K-1+T)+K-1+t]; }
  wf(st,(K-1)*C); wf(x,T*C); wf((float*)w->data,K*C); wf((float*)o->data,T*C);
  ggml_free(c);
}
int main(int argc,char**argv){
  ggml_cpu_init();
  f=fopen(argv[1],"wb");
  int n=0; long pos0=ftell(f); wi(0);
  int s0[4]={0,0,0,0}, s1[4]={3,2,2,1}, s2[4]={2,3,3,0}, s3[4]={4,2,2,0};
  // GGML_ROPE_TYPE_NORMAL 0, NEOX 2, MROPE 8, IMROPE 40
  rope_case(0,16,s0,1.f,5,3,16); rope_case(2,16,s0,1.f,5,3,16); rope_case(2,8,s0,0.5f,5,3,16);
  rope_case(8,16,s2,1.f,6,2,16); rope_case(40,16,s1,1.f,6,2,16); rope_case(40,16,s3,1.f,6,2,16); rope_case(40,32,(int[]){11,11,10,0},1.f,4,2,64);
  delta_case(5,2,4,16); delta_case(3,2,2,32); delta_case(4,1,3,128);
  conv_case(7,24,4); conv_case(1,24,4); conv_case(3,8,5);
  fclose(f);
  return 0;
}
