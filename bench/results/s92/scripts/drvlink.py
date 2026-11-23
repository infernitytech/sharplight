# usage: drvlink.py <in.ptx> <out.cubin> : the driver's JIT compilation of a PTX file (what the engine loads without NVRTC SASS)
import ctypes, sys
cu = ctypes.CDLL("libcuda.so.1")
def ck(r, what):
    if r: raise SystemExit(f"{what} failed {r}")
ck(cu.cuInit(0), "init"); dev = ctypes.c_int(); ck(cu.cuDeviceGet(ctypes.byref(dev), 0), "dev")
ctx = ctypes.c_void_p(); ck(cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "retain"); ck(cu.cuCtxSetCurrent(ctx), "set")
data = open(sys.argv[1], "rb").read()
data = data.rstrip(b"\0") + b"\0"
st = ctypes.c_void_p()
import os
tgt = os.environ.get("JIT_TARGET")
if tgt:
    opts = (ctypes.c_int * 1)(9); vals = (ctypes.c_void_p * 1)(int(tgt))
    ck(cu.cuLinkCreate_v2(1, opts, vals, ctypes.byref(st)), "linkcreate target")
else:
    ck(cu.cuLinkCreate_v2(0, None, None, ctypes.byref(st)), "linkcreate")
ck(cu.cuLinkAddData_v2(st, 1, data, len(data), b"k", 0, None, None), "adddata")  # CU_JIT_INPUT_PTX = 1
out = ctypes.c_void_p(); size = ctypes.c_size_t()
ck(cu.cuLinkComplete(st, ctypes.byref(out), ctypes.byref(size)), "complete")
open(sys.argv[2], "wb").write(ctypes.string_at(out, size.value))
