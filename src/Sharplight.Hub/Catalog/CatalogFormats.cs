namespace Sharplight.Hub.Catalog;

/// <summary>The <see cref="CatalogVariant.Format"/> values.</summary>
public static class CatalogFormats
{
    /// <summary>GGUF file(s).</summary>
    public const string Gguf = "gguf";

    /// <summary>A Hugging Face checkpoint folder with safetensors weights.</summary>
    public const string Safetensors = "safetensors";

    /// <summary>A checkpoint folder with PyTorch weights (<c>.bin</c>, <c>.pth</c>, <c>.pt</c>).</summary>
    public const string PyTorch = "pytorch";

    /// <summary>A diffusers pipeline folder.</summary>
    public const string Diffusers = "diffusers";

    /// <summary>A single-file diffusion checkpoint (<c>.safetensors</c> / <c>.ckpt</c>).</summary>
    public const string SingleFile = "single-file";

    /// <summary>A whisper.cpp <c>.bin</c> file.</summary>
    public const string WhisperCpp = "whisper-cpp";

    /// <summary>ONNX graph(s).</summary>
    public const string Onnx = "onnx";

    /// <summary>A Sharplight package.</summary>
    public const string Slm = "slm";

    /// <summary>Every known format.</summary>
    public static IReadOnlyList<string> All { get; } = [Gguf, Safetensors, PyTorch, Diffusers, SingleFile, WhisperCpp, Onnx, Slm];

    /// <summary>The store format of a catalog format (what <see cref="ModelManifest.Format"/> records).</summary>
    /// <param name="format">One of <see cref="All"/>.</param>
    public static ModelFormat ToModelFormat(string format) => format switch
    {
        Gguf => ModelFormat.Gguf,
        Diffusers or SingleFile => ModelFormat.Diffusers,
        WhisperCpp => ModelFormat.WhisperCpp,
        Onnx => ModelFormat.Onnx,
        Slm => ModelFormat.Slm,
        _ => ModelFormat.Safetensors,
    };
}
