using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>
    /// True when the fp8 tensor-core GEMM (<c>gemm_fp8.slang</c>, <see cref="DeviceFeatures.MatrixFp8"/>) serves
    /// prefill-sized batches of a matmul with block-scaled E4M3 weights: activations quantized to E4M3 with one scale per
    /// 128 values, f8 x f8 MMAs with f32 accumulation, one scaling per weight block. Without the feature (or with
    /// <c>SHARPLIGHT_FP8_GEMM=0</c>) the f16 tensor-core GEMM decodes the weights instead.
    /// </summary>
    private bool UseFp8Gemm(DataType type, long k)
        => DiagnosticSwitches.Fp8Gemm && _opt.Has(DeviceFeatures.MatrixFp8) && _opt.Has(DeviceFeatures.SubgroupOps) && _opt.SubgroupSize == 32
           && (type == DataType.F8E4M3_B128 || type == DataType.F8E4M3_B32) && k % 128 == 0;

    /// <summary>Emits the E4M3 activation quantization and the fp8 GEMM launches of a planned matmul (see <see cref="UseFp8Gemm"/>).</summary>
    private void EmitFp8Gemm(MatMulPlan p) => EmitQuantizedGemm(p, "fp8", ActivationCode.Fp8, 128,
        (tileRows, split) => Lib("gemm_fp8.slang", "gemm_fp8", Fp8GemmMacros(p.W.Type, p.N, p.K, tileRows, split)));

    private Dictionary<string, string> Fp8GemmMacros(DataType type, long n, long k, int tileRows, int split)
    {
        var layout = MmqLayout(tileRows);
        if (layout[0] * layout[2] * 16 != tileRows)
        {
            throw new NotSupportedException($"fp8 GEMM layout {string.Join(',', layout)} does not give {tileRows}-token tiles.");
        }

        return Macros(("KERNEL_GEMM_FP8", 1), ("WBLOCK", type.BlockSize), ("K", k), ("N", n), ("Y_STRIDE", n), ("ROW_BYTES", type.ByteSize(k) + "ull"),
            ("F8_WM", layout[0]), ("F8_WN", layout[1]), ("F8_FM", layout[2]), ("F8_FN", layout[3]), ("SPLITK", split),
            ("F8_GENERIC_LAYOUT", DiagnosticSwitches.MmqGenericLayout));
    }
}
