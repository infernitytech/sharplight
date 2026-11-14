Operator schemas of onnx 1.23.2 (ai.onnx opset 28) and onnxruntime's com.microsoft contrib schemas.

| domain | operators | importer | cpu: all tests pass | vulkan: all tests pass | vulkan: without CPU fallback |
|---|---|---|---|---|---|
| ai.onnx | 203 | 201 | 161 / 192 with tests | 161 / 192 with tests | 147 |
| ai.onnx.ml | 19 | 16 | 16 / 16 with tests | 16 / 16 with tests | 9 |
| ai.onnx.preview | 1 | 1 | 1 / 1 with tests | 1 / 1 with tests | 1 |
| ai.onnx.preview.training | 4 | 3 | 3 / 3 with tests | 3 / 3 with tests | 3 |
| com.microsoft | 136 | 53 | 45 / 47 with tests | 45 / 47 with tests | 44 |
| com.microsoft (common set) | 61 | 52 | 44 / 46 with tests | 44 / 46 with tests | 43 |

ONNX backend node tests:

| provider | tests | pass | pass without CPU fallback | wrong | import fail | compile fail | run fail | harness unsupported |
|---|---|---|---|---|---|---|---|---|
| cpu | 1884 | 1829 (97.1%) | 1829 (97.1%) | 8 | 47 | 0 | 0 | 0 |
| vulkan | 1884 | 1829 (97.1%) | 1745 (92.6%) | 8 | 47 | 0 | 0 | 0 |

Extra tests (ai.onnx.ml, strings, signal, com.microsoft; onnxruntime outputs):

| provider | tests | pass | pass without CPU fallback | wrong | import fail | compile fail | run fail | harness unsupported |
|---|---|---|---|---|---|---|---|---|
| cpu | 172 | 170 (98.8%) | 170 (98.8%) | 0 | 2 | 0 | 0 | 0 |
| vulkan | 172 | 170 (98.8%) | 127 (73.8%) | 0 | 2 | 0 | 0 | 0 |


### ai.onnx

| operator | versions | importer | cpu tests | vulkan tests | notes |
|---|---|---|---|---|---|
