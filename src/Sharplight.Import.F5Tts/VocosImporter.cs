using System.Text.RegularExpressions;
using Sharplight.Import.HuggingFace;
using Sharplight.Models;

namespace Sharplight.Import.F5Tts;

/// <summary>
/// Vocos importer: maps a Vocos mel vocoder folder (<c>config.yaml</c> naming <c>VocosBackbone</c> and <c>ISTFTHead</c>, <c>pytorch_model.bin</c>) onto the
/// canonical <see cref="ModelSource"/> of architecture <c>vocos</c>, an auxiliary architecture: it is loaded as the <c>vocoder</c> of a model that
/// generates mel spectrograms (F5-TTS). The weights keep their checkpoint names (<c>backbone.*</c>, <c>head.*</c>); the hyper-parameters
/// (<c>vocos.*</c>: mel bins, layers, FFT size, hop) come from the configuration.
/// </summary>
public sealed partial class VocosImporter : IModelImporter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "vocos";

    /// <inheritdoc/>
    public string Format => "vocos";

    /// <inheritdoc/>
    public int Priority => 100;

    /// <inheritdoc/>
    public bool CanOpen(string path)
        => Directory.Exists(path) && File.Exists(Path.Combine(path, "pytorch_model.bin")) && File.Exists(Path.Combine(path, "config.yaml"))
           && File.ReadAllText(Path.Combine(path, "config.yaml")).Contains("VocosBackbone", StringComparison.Ordinal);

    /// <inheritdoc/>
    ModelSource IModelImporter.Open(string path) => Open(path);

    /// <summary>Opens a Vocos folder.</summary>
    /// <param name="path">The folder.</param>
    public static ModelSource Open(string path)
    {
        string config = File.ReadAllText(Path.Combine(path, "config.yaml"));
        var file = SafetensorsFile.OpenTorch(Path.Combine(path, "pytorch_model.bin"));
        var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        foreach (var (name, tensor) in file.Tensors)
        {
            if (tensor.Type is not null && !name.StartsWith("feature_extractor.", StringComparison.Ordinal) && !name.EndsWith(".window", StringComparison.Ordinal))
            {
                weights[name] = tensor.ToWeightInfo().As(name);
            }
        }

        const string A = Architecture + ".";
        return new ModelSource
        {
            Format = "vocos",
            Architecture = Architecture,
            Metadata = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                [CanonicalKeys.Architecture] = Architecture,
                [CanonicalKeys.Name] = Path.GetFileName(path.TrimEnd(Path.DirectorySeparatorChar)),
                [A + "mel_bins"] = Setting(config, "input_channels", 100),
                [A + "layers"] = Setting(config, "num_layers", 8),
                [A + "fft_size"] = Setting(config, "n_fft", 1024),
                [A + "hop_length"] = Setting(config, "hop_length", 256),
                [A + "sample_rate"] = Setting(config, "sample_rate", 24000),
            },
            Weights = weights,
            OwnedResources = [file],
            Path = path,
        };
    }

    // The first "key: number" in the YAML text.
    private static int Setting(string yaml, string key, int fallback)
    {
        var match = Regex.Match(yaml, $@"^\s*{Regex.Escape(key)}:\s*(\d+)\s*$", RegexOptions.Multiline);
        return match.Success ? int.Parse(match.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture) : fallback;
    }
}
