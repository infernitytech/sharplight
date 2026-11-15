# source this: activates the ORT venv (onnxruntime-gpu + CUDA 13 / cuDNN 9) and puts the CUDA (and, if present, TensorRT) libraries on
# LD_LIBRARY_PATH. Override the locations with ORT_VENV, CUDA_LIB and TRT_LIB.
ORT_VENV=${ORT_VENV:-~/venvs/ort}
source "$ORT_VENV/bin/activate"
export ORT_PYTHON="$ORT_VENV/bin/python3"
SP=$(echo "$ORT_VENV"/lib/python3*/site-packages)
export LD_LIBRARY_PATH=$SP/nvidia/cu13/lib:$SP/nvidia/cudnn/lib:${CUDA_LIB:-/usr/local/cuda/lib64}${TRT_LIB:+:$TRT_LIB}:$LD_LIBRARY_PATH
