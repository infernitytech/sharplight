namespace Sharplight.Studio.Services.Runtime;

/// <summary>One LoRA adapter merged into an image model at load.</summary>
/// <param name="Path">The LoRA <c>.safetensors</c> file.</param>
/// <param name="Scale">The strength (1 = as trained).</param>
internal sealed record ModelLora(string Path, float Scale)
{
    /// <summary>The file name without extension, as the adapter is shown.</summary>
    public string Name => System.IO.Path.GetFileNameWithoutExtension(Path);
}
