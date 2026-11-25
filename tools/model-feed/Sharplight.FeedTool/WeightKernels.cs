using Sharplight.ModelFeed;
using Sharplight.Runtime;

namespace Sharplight.FeedTool;

/// <summary>
/// The device features whose kernels accelerate each engine weight type (docs/model-support.md "Kernels", docs/cuda.md): advisory for
/// hosts ranking devices. Every weight type the engine stores has a generic kernel on every provider (a lowering selects the faster
/// kernels only through device features and keeps the generated kernel as the fallback), so <see cref="PackageKernel.Fallback"/> is set.
/// </summary>
internal static class WeightKernels
{
    private static readonly string[] s_fp8 = [DeviceFeatures.NativeGemmFp8, DeviceFeatures.MatrixFp8, DeviceFeatures.Float8, DeviceFeatures.MatrixF16];
    private static readonly string[] s_mxfp4 = [DeviceFeatures.NativeGemmMoe, DeviceFeatures.MatrixF16];
    private static readonly string[] s_integerDot = [DeviceFeatures.IntDot, DeviceFeatures.MatrixF16];
    private static readonly string[] s_half = [DeviceFeatures.NativeGemmF16, DeviceFeatures.LibraryGemm, DeviceFeatures.MatrixF16, DeviceFeatures.MatrixBF16];
    private static readonly string[] s_block = [DeviceFeatures.MatrixF16];

    /// <summary>The kernel entries of a package's weight types (largest share first); integer and boolean tensors are left out.</summary>
    /// <param name="weightTypes">Weight bytes per type.</param>
    /// <returns>The entries.</returns>
    public static IReadOnlyList<PackageKernel> For(IEnumerable<KeyValuePair<string, long>> weightTypes) =>
        [.. weightTypes.Where(pair => Features(pair.Key) is not null).Select(pair => new PackageKernel { WeightType = pair.Key, Bytes = pair.Value, NativeFeatures = Features(pair.Key)! })];

    private static string[]? Features(string type) => type switch
    {
        _ when type.StartsWith("f8e", StringComparison.Ordinal) => s_fp8,
        "mxfp4" => s_mxfp4,
        _ when type.StartsWith("nvfp4", StringComparison.Ordinal) || type.StartsWith("q4z_", StringComparison.Ordinal) || type.StartsWith("q8z_", StringComparison.Ordinal) => s_integerDot,
        "f16" or "bf16" => s_half,
        "f32" or "f64" => [],
        _ when type.StartsWith('i') && !type.StartsWith("iq", StringComparison.Ordinal) => null,
        "u8" or "u32" or "bool" or "str" => null,
        _ => s_block,
    };
}
