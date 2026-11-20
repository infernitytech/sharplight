# Compiles a .cu file with the provider's NVRTC and its options into PTX. usage: nvrtc_ptx.py src.cu out.ptx [nvrtc options]; env NVRTC_DIR
import ctypes, sys
import os
lib = ctypes.CDLL(os.path.join(os.environ["NVRTC_DIR"], "libnvrtc.so.13"))
src = open(sys.argv[1], "rb").read()
prog = ctypes.c_void_p()
assert lib.nvrtcCreateProgram(ctypes.byref(prog), src, b"k.cu", 0, None, None) == 0
opts = [b"-arch=compute_120", b"-default-device", b"-std=c++17", b"-w"] + [a.encode() for a in sys.argv[3:]]
arr = (ctypes.c_char_p * len(opts))(*opts)
r = lib.nvrtcCompileProgram(prog, len(opts), arr)
n = ctypes.c_size_t(); lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n)); log = ctypes.create_string_buffer(n.value); lib.nvrtcGetProgramLog(prog, log)
if r != 0:
    print(log.value.decode()); sys.exit(1)
lib.nvrtcGetPTXSize(prog, ctypes.byref(n)); ptx = ctypes.create_string_buffer(n.value); lib.nvrtcGetPTX(prog, ptx)
open(sys.argv[2], "wb").write(ptx.value)
