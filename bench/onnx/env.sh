# source this: activates the ORT venv (onnxruntime-gpu 1.31 + CUDA 13 / cuDNN 9 pip wheels)
source ~/venvs/ort/bin/activate
SP=~/venvs/ort/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SP/nvidia/cu13/lib:$SP/nvidia/cudnn/lib:$LD_LIBRARY_PATH
