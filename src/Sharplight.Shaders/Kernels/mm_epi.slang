// Store epilogue of the weight matmuls (matmul.slang, mmvq.slang), included after the kernel's Params. With MM_EPI the including
// generated source defines `float mm_epi(float m0, float a0, float a1, float a2)`, an elementwise group applied to each result m0
// in the store, which reads up to three extra tensors (Params e0 .. e2, MM_EPI_ARGS of them, access kinds MM_EPI_K0 .. MM_EPI_K2);
// y is the group's output, dense with Y_STRIDE = N. MM_STORE(v, t, n) is the value stored for product v of output (t, n).
#ifndef MM_EPI
#define MM_EPI 0
#endif
#if MM_EPI
// Element of epilogue tensor e for output (t, n): kind 0 one scalar, 1 a row vector over the output columns, 2 a dense
// tensor of the output's shape.
float mm_epi_arg(float* e, uint kind, uint64_t t, uint n)
{
    return kind == 0 ? e[0] : kind == 1 ? e[n] : e[t * N + n];
}

// The product v of output (t, n) through the generated epilogue mm_epi (the fused elementwise group).
float mm_store_value(float v, Params p, uint64_t t, uint n)
{
    float a0 = 0, a1 = 0, a2 = 0;
#if MM_EPI_ARGS > 0
    a0 = mm_epi_arg(p.e0, MM_EPI_K0, t, n);
#endif
#if MM_EPI_ARGS > 1
    a1 = mm_epi_arg(p.e1, MM_EPI_K1, t, n);
#endif
#if MM_EPI_ARGS > 2
    a2 = mm_epi_arg(p.e2, MM_EPI_K2, t, n);
#endif
    return mm_epi(v, a0, a1, a2);
}
#define MM_STORE(v, t, n) mm_store_value(v, p, t, n)
#else
#define MM_STORE(v, t, n) (v)
#endif
