using Sharplight.Hub;

namespace Sharplight.Studio.Models;

/// <summary>The names Studio shows for <see cref="ModelFormat"/> values.</summary>
internal static class ModelFormatTitles
{
    /// <summary>The name of a format in badges and details.</summary>
    /// <param name="format">The format.</param>
    public static string Title(ModelFormat format) => format switch
    {
        ModelFormat.Gguf => "GGUF",
        ModelFormat.Safetensors => "Safetensors",
        ModelFormat.Onnx => "ONNX",
        ModelFormat.Diffusers => "Diffusers",
        ModelFormat.WhisperCpp => "whisper.cpp",
        ModelFormat.Slm => ".slm",
        _ => format.ToString(),
    };
}
