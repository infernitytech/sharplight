bench/s68_f16dec (gemm_f16_native.cu H_CFG 2), gpt-oss-20b shapes, T = 64 rows unless noted; "isolated": best of 30 after an L2 flush;
"sustained": 24 distinct weight copies back to back (us per launch).
f16_decode_variants.txt: qkv64 / qkv32 = q / k / v multi-output on 64 / 32-row weight tiles; q_s6 = six stages; q_pf / q_pf256 =
cp.async with L2::128B / L2::256B prefetch; o32 / o64 = o projection [4096 -> 2880] on 32 / 64-row weight tiles (likewise _s6, _pf).
First session (same binaries): qkv64 39.4 sustained (47.1 isolated), qkv32 39.0, o64 33.3, o32 32.0-32.1; T=32: qkv64 46.1, o64 36.7.
../decode/fp8_tile_rows.txt: bench/s57_gemm with gemm_fp8_native.cu, F8N_BN=64, F8N_BM=64 (four stages) vs 32 (three stages), at
16 / 32 rows (its "sustained" repeats one weight copy, which stays in L2 for the small shapes: use the flushed line).
