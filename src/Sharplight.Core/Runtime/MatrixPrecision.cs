namespace Sharplight.Runtime;

/// <summary>
/// Precision of the products in f32 matrix multiplications, convolutions and attention (<see cref="CompileOptions.MatrixPrecision"/>).
/// Sums are always accumulated in f32; the setting only decides how many significant bits of each input enter a product.
/// </summary>
public enum MatrixPrecision
{
    /// <summary>
    /// The default: products as accurate as f32 arithmetic. On devices with f16 matrix units the inputs are split into two
    /// f16 parts and each product is formed from three tensor-core products (see <see cref="CompileOptions.SplitHalfMatMul"/>).
    /// </summary>
    Strict,

    /// <summary>
    /// Opt-in: one tensor-core product per multiply-add of 11-significant-bit f16 inputs, the accuracy class of TF32 (the
    /// default of onnxruntime's CUDA execution provider and of PyTorch convolutions). Rows of the activations and weights are
    /// scaled by powers of two before rounding, so values keep f32's range (no f16 overflow or subnormals); attention runs its
    /// f16 kernels. Relative errors are about 1e-4 to 1e-3 instead of 1e-7. Devices without f16 matrix units keep the
    /// strict kernels.
    /// </summary>
    Fast,
}
