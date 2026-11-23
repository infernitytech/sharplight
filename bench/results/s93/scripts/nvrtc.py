# nvrtc.py src.cu out.(ptx|cubin) [-DX=Y ...]: compile with the engine's NVRTC 13.0 (options of NvrtcTranslator.Options)
import ctypes, sys
lib = ctypes.CDLL('~/sl-worktrees/perf-r35/src/Sharplight.Cuda.Runtime/bin/Release/net11.0/libnvrtc.so.13')
src, out = sys.argv[1], sys.argv[2]
cubin = out.endswith('.cubin')
opts = [('-arch=sm_120f' if cubin else '-arch=compute_120f'), '-default-device', '-std=c++17', '-w'] + sys.argv[3:]
prog = ctypes.c_void_p()
code = open(src, 'rb').read()
r = lib.nvrtcCreateProgram(ctypes.byref(prog), code, b'k.cu', 0, None, None)
arr = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
r = lib.nvrtcCompileProgram(prog, len(opts), arr)
n = ctypes.c_size_t()
lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n)); log = ctypes.create_string_buffer(n.value); lib.nvrtcGetProgramLog(prog, log)
if r != 0:
    print('compile failed', r, log.value.decode()[:3000]); sys.exit(1)
if cubin:
    lib.nvrtcGetCUBINSize(prog, ctypes.byref(n)); b = ctypes.create_string_buffer(n.value); lib.nvrtcGetCUBIN(prog, b)
else:
    lib.nvrtcGetPTXSize(prog, ctypes.byref(n)); b = ctypes.create_string_buffer(n.value); lib.nvrtcGetPTX(prog, b)
open(out, 'wb').write(b.raw[:n.value])
