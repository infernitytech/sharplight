A/B files: lines "r<round> <A|B> <model> <--bench output>"; A = default, B = the change
(mvrows: A = SHARPLIGHT_MV_ROWS=1, B = default; both: A = develop 282dd5b, B = both opt-ins).
greedy_identity.txt: outputs per variant against "off" (SHARPLIGHT_MV_ROWS=1 SHARPLIGHT_ATTN_FE_CAUSAL=0 SHARPLIGHT_GEMM_WAVE=192):
on = all three changes, v_mv / v_fe / v_wave = only that change, def = the final defaults.
