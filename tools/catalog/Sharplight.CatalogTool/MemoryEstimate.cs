using Sharplight.Hub.Catalog;

namespace Sharplight.CatalogTool;

/// <summary>
/// The rough device memory a variant needs: its weights (with companions) plus a share for activations, the KV cache of a short
/// context and the runtime. It only ranks variants against a device ("fits your GPU"); the loader measures the real need.
/// </summary>
internal static class MemoryEstimate
{
    private const double ActivationShare = 0.15;
    private const long RuntimeBytes = 384L << 20;
    private static readonly string[] s_weightExtensions = [".gguf", ".safetensors", ".bin", ".pth", ".pt", ".ckpt", ".onnx", ".slm", ".onnx_data"];

    /// <summary>The estimate in bytes for <paramref name="variant"/> of <paramref name="entry"/>.</summary>
    public static long For(CatalogEntry entry, CatalogVariant variant)
    {
        long weights = WeightBytes(variant) + entry.CompanionsOf(variant).Sum(WeightBytes);
        return weights == 0 ? 0 : (long)(weights * (1 + ActivationShare)) + RuntimeBytes;
    }

    private static long WeightBytes(CatalogFileSet set)
        => set.Files.Where(file => s_weightExtensions.Any(extension => file.Path.EndsWith(extension, StringComparison.OrdinalIgnoreCase))).Sum(file => file.Size);
}
