#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
void launch_q(const float*, unsigned char*, unsigned);
void launch_g(const unsigned char*, const unsigned char*, float*, unsigned, cudaStream_t);
static float e2m1(int c){ static const float m[8]={0,0.5f,1,1.5f,2,3,4,6}; return (c&8)?-m[c&7]:m[c&7]; }
static float e4m3(int c){ int e=(c>>3)&15, m=c&7; return e==0? m/512.f : ldexpf(1+m/8.f,e-7); }
__global__ void rdflush(const int4* p, size_t n, int* o){ int4 a{0,0,0,0}; for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){int4 v=p[i]; a.x^=v.x;a.y^=v.y;} if(a.x==12345&&a.y==777) *o=1; }
int main(int argc,char**argv){
  int T=atoi(argv[1]); int iters=argc>2?atoi(argv[2]):0;
  const int Kv=K, Nv=N; const int WBB = F4_GS?40:36; size_t WRB=(size_t)Kv/64*WBB; size_t XRB=(size_t)Kv/256*160+32;
  std::mt19937 rng(5); std::normal_distribution<float> nd(0,1); std::uniform_int_distribution<int> ui(0,255);
  std::vector<float> x((size_t)T*Kv); for(auto&v:x) v=nd(rng)*(1+ (rng()%7==0?5:0));
  int NW = DUAL?2:1; std::vector<std::vector<unsigned char>> w(NW, std::vector<unsigned char>(WRB*Nv));
  for(int m=0;m<NW;m++) for(int n=0;n<Nv;n++){ float g = 0.01f*(1+n%3); for(int b=0;b<Kv/64;b++){ unsigned char* blk=&w[m][n*WRB+b*WBB]; int o=0; if(F4_GS){memcpy(blk,&g,4);o=4;} for(int s=0;s<4;s++) blk[o+s]=0x30+ui(rng)%24; for(int i=0;i<32;i++) blk[o+4+i]=ui(rng);} }
  float *dx,*dy; unsigned char *dq,*dw,*dw2;
  cudaMalloc(&dx,x.size()*4); cudaMalloc(&dq,XRB*T); cudaMalloc(&dw,WRB*Nv*NW); size_t ys=(size_t)T*Y_STRIDE; cudaMalloc(&dy,ys*4);
  cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice);
  for(int m=0;m<NW;m++) cudaMemcpy(dw+m*WRB*Nv,w[m].data(),WRB*Nv,cudaMemcpyHostToDevice);
  launch_q(dx,dq,T);
  std::vector<unsigned char> q(XRB*T); cudaMemcpy(q.data(),dq,q.size(),cudaMemcpyDeviceToHost);
  // dequantized activations and quantization error
  std::vector<double> xd((size_t)T*Kv); double qe=0,qn=0;
  for(int t=0;t<T;t++){ unsigned char* r=&q[t*XRB]; float gx; memcpy(&gx,r+(Kv/256)*160,4);
    for(int k=0;k<Kv;k++){ int st=k/256, gi=(k%256)/16, j=k%16; unsigned char* rec=r+st*160; int byte=rec[gi*8+(j%8)]; int code = j<8? byte&15 : byte>>4; xd[(size_t)t*Kv+k]=e2m1(code)*e4m3(rec[128+gi])*gx; double d=xd[(size_t)t*Kv+k]-x[(size_t)t*Kv+k]; qe+=d*d; qn+=x[(size_t)t*Kv+k]*(double)x[(size_t)t*Kv+k]; } }
  printf("activation quant rel L2 err %.4f\n", sqrt(qe/qn));
  // reference GEMM using dequantized x
  std::vector<std::vector<double>> wd(NW, std::vector<double>((size_t)Nv*Kv));
  for(int m=0;m<NW;m++) for(int n=0;n<Nv;n++){ for(int b=0;b<Kv/64;b++){ unsigned char* blk=&w[m][n*WRB+b*WBB]; float g=1; int o=0; if(F4_GS){memcpy(&g,blk,4);o=4;} for(int s=0;s<4;s++) for(int j=0;j<16;j++){ int byte=blk[o+4+s*8+(j%8)]; int code=j<8?byte&15:byte>>4; wd[m][(size_t)n*Kv+b*64+s*16+j]=e2m1(code)*e4m3(blk[o+s])*g; } } }
  launch_g(dq,dw,dy,T,0); cudaError_t e=cudaDeviceSynchronize(); if(e){printf("err %s\n",cudaGetErrorString(e));return 1;}
  std::vector<float> y(ys); cudaMemcpy(y.data(),dy,ys*4,cudaMemcpyDeviceToHost);
  double err=0,nrm=0,terr=0,tnrm=0; int checkT = T<64?T:64;
  for(int t=0;t<checkT;t++) for(int n=0;n<(DUAL?Nv:Nv);n++){ double r0=0,r1=0; for(int k=0;k<Kv;k++){ r0+=xd[(size_t)t*Kv+k]*wd[0][(size_t)n*Kv+k]; if(DUAL) r1+=xd[(size_t)t*Kv+k]*wd[1][(size_t)n*Kv+k]; }
    double ref = DUAL ? (r0/(1+exp(-r0)))*r1 : r0; double d=y[(size_t)t*Y_STRIDE+n]-ref; err+=d*d; nrm+=ref*ref;
    if(!DUAL){ double tr=0; for(int k=0;k<Kv;k++) tr+=x[(size_t)t*Kv+k]*wd[0][(size_t)n*Kv+k]; double dt=y[(size_t)t*Y_STRIDE+n]-tr; terr+=dt*dt; tnrm+=tr*tr; } }
  printf("gemm rel L2 err vs dequant ref %.3e, vs exact x %.3e\n", sqrt(err/nrm), sqrt(terr/tnrm));
  if(iters){ cudaEvent_t a,b; cudaEventCreate(&a);cudaEventCreate(&b); for(int i=0;i<5;i++) launch_g(dq,dw,dy,T,0);
    void* fl; cudaMalloc(&fl,256<<20); cudaMemset(fl,0,256<<20); float best=1e9; for(int r=0;r<iters;r++){ rdflush<<<328,512>>>((const int4*)fl,(256<<20)/16,(int*)fl); cudaEventRecord(a); launch_g(dq,dw,dy,T,0); cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms,a,b); if(ms<best)best=ms;}
    double bytes = (double)WRB*Nv*NW + (double)XRB*T; printf("T=%d best %.1f us  %.0f GB/s  %.1f TFLOPS\n",T,best*1e3,bytes/best/1e6, 2.0*T*Nv*NW*(double)Kv/best/1e9); }
}
