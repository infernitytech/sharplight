using Sharplight.Hub;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>A format choice of the search filter.</summary>
/// <param name="Name">Shown in the picker.</param>
/// <param name="Format">The format, or null for any.</param>
internal sealed record FormatFilter(string Name, ModelFormat? Format)
{
    /// <summary>The choices.</summary>
    public static IReadOnlyList<FormatFilter> All { get; } =
        [new("Any format", null), new("GGUF", ModelFormat.Gguf), new("Safetensors", ModelFormat.Safetensors), new("ONNX", ModelFormat.Onnx)];
}
