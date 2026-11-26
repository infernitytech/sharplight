using System.Text.Json;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// A diffusers model folder of one stand-alone component (as published on its own: <c>config.json</c> and
/// <c>diffusion_pytorch_model.safetensors</c>), such as a tiny autoencoder (<c>AutoencoderTiny</c>) or a ControlNet (<c>ControlNetModel</c>):
/// it opens as its weights file, whose reader takes the configuration beside it.
/// </summary>
internal static class ComponentFolder
{
    private static readonly string[] s_classes = ["AutoencoderTiny", "ControlNetModel"];

    /// <summary>The weights file of a component folder, or null when <paramref name="path"/> is not one.</summary>
    /// <param name="path">A directory.</param>
    public static string? WeightsFile(string path)
    {
        string config = Path.Combine(path, "config.json");
        if (!Directory.Exists(path) || !File.Exists(config) || File.Exists(Path.Combine(path, "model_index.json")))
        {
            return null;
        }

        try
        {
            using var document = JsonDocument.Parse(File.ReadAllBytes(config));
            if (!document.RootElement.TryGetProperty("_class_name", out var name) || name.ValueKind != JsonValueKind.String
                || !s_classes.Contains(name.GetString()))
            {
                return null;
            }
        }
        catch (JsonException)
        {
            return null;
        }

        var files = Directory.GetFiles(path, "*.safetensors").Order(StringComparer.Ordinal).ToList();
        // Full-precision weights are preferred over the .fp16 variant when both are there.
        return files.FirstOrDefault(f => !Path.GetFileName(f).Contains(".fp16.", StringComparison.Ordinal)) ?? files.FirstOrDefault();
    }
}
