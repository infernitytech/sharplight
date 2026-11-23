extern "C" __global__ void t(float2* p, float2 a, float2 b)
{
    float2 c = p[threadIdx.x];
    unsigned long long r, x = *(unsigned long long*)&a, y = *(unsigned long long*)&b, z = *(unsigned long long*)&c;
    asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(r) : "l"(x), "l"(y), "l"(z));
    p[threadIdx.x] = *(float2*)&r;
}
