#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
__global__ void k(const uint32_t* A, const uint32_t* B, const uint32_t* SA, const uint32_t* SB, float* D) {
  int l = threadIdx.x;
  uint32_t a0=A[l*4+0],a1=A[l*4+1],a2=A[l*4+2],a3=A[l*4+3];
  uint32_t b0=B[l*2+0],b1=B[l*2+1];
  float d0=0,d1=0,d2=0,d3=0;
  uint32_t sa=SA[l], sb=SB[l];
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{0,0},%11,{0,0};"
    : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb));
  D[l*4+0]=d0;D[l*4+1]=d1;D[l*4+2]=d2;D[l*4+3]=d3;
}
static float e2m1(int c){ static const float m[8]={0,0.5f,1,1.5f,2,3,4,6}; return (c&8)?-m[c&7]:m[c&7]; }
static float e4m3(uint8_t c){ int e=(c>>3)&15, m=c&7; float v = e==0 ? ldexpf(m/8.f,-6) : ldexpf(1+m/8.f,e-7); return (c&0x80)?-v:v; }
int main(){
  srand(1);
  int Am[16][64], Bm[8][64]; uint8_t Sa[16][4], Sb[8][4];
  for(int r=0;r<16;r++)for(int k=0;k<64;k++)Am[r][k]=rand()&15;
  for(int r=0;r<8;r++)for(int k=0;k<64;k++)Bm[r][k]=rand()&15;
  for(int r=0;r<16;r++)for(int j=0;j<4;j++)Sa[r][j]=0x30+(rand()%24);
  for(int r=0;r<8;r++)for(int j=0;j<4;j++)Sb[r][j]=0x30+(rand()%24);
  uint32_t hA[128]={0},hB[64]={0},hSA[32]={0},hSB[32]={0}; float hD[128];
  for(int l=0;l<32;l++){int g=l/4,c=l%4;
    for(int reg=0;reg<4;reg++){int row=g+((reg&1)?8:0); int k0=c*8+((reg&2)?32:0); uint32_t v=0; for(int i=0;i<8;i++) v|=(uint32_t)Am[row][k0+i]<<(4*i); hA[l*4+reg]=v;}
    for(int reg=0;reg<2;reg++){int k0=c*8+(reg?32:0); uint32_t v=0; for(int i=0;i<8;i++) v|=(uint32_t)Bm[g][k0+i]<<(4*i); hB[l*2+reg]=v;}
    if(c==0){ for(int j=0;j<4;j++){ hSA[l]|=(uint32_t)Sa[g][j]<<(8*j); hSB[l]|=(uint32_t)Sb[g][j]<<(8*j);} }
    if(c==1){ for(int j=0;j<4;j++) hSA[l]|=(uint32_t)Sa[g+8][j]<<(8*j); }
  }
  uint32_t *dA,*dB,*dSA,*dSB; float* dD;
  cudaMalloc(&dA,512);cudaMalloc(&dB,256);cudaMalloc(&dSA,128);cudaMalloc(&dSB,128);cudaMalloc(&dD,512);
  cudaMemcpy(dA,hA,512,cudaMemcpyHostToDevice);cudaMemcpy(dB,hB,256,cudaMemcpyHostToDevice);
  cudaMemcpy(dSA,hSA,128,cudaMemcpyHostToDevice);cudaMemcpy(dSB,hSB,128,cudaMemcpyHostToDevice);
  k<<<1,32>>>(dA,dB,dSA,dSB,dD); cudaMemcpy(hD,dD,512,cudaMemcpyDeviceToHost);
  double maxerr=0;
  for(int l=0;l<32;l++){int g=l/4,c=l%4; for(int q=0;q<4;q++){int row=g+(q>=2?8:0), col=2*c+(q&1);
    double ref=0; for(int k=0;k<64;k++) ref+= (double)e2m1(Am[row][k])*e2m1(Bm[col][k])*e4m3(Sa[row][k/16])*e4m3(Sb[col][k/16]);
    maxerr=fmax(maxerr,fabs(ref-hD[l*4+q])/(fabs(ref)+1e-3)); if(l<2)printf("%f %f\n",ref,hD[l*4+q]);}}
  printf("max rel err %g\n",maxerr);
}
