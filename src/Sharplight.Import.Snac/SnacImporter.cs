using System.Text.Json;
using Sharplight.Import.PyTorch;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.Snac;

/// <summary>
/// SNAC importer: maps a SNAC checkpoint directory (<c>config.json</c>, <c>pytorch_model.bin</c>) onto the canonical <see cref="ModelSource"/> of architecture
/// <c>snac</c>, the decoding half of the codec only (the quantizer's codebooks and projections, the decoder): weight-normalized convolutions (stored as the
/// parametrization's <c>original0</c> gain and <c>original1</c> direction) are folded into plain <c>weight</c>s on first read, the snake parameters are
/// flattened to <c>[channels]</c>, and the encoder is not imported.
/// </summary>
/// <remarks>Register with <c>services.AddSnacImporter()</c> so that the <see cref="ModelLoader"/> opens SNAC checkpoints, or call <see cref="Open"/>.</remarks>
public sealed class SnacImporter : IModelImporter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "snac";

    private const string Gain = ".parametrizations.weight.original0";
    private const string Direction = ".parametrizations.weight.original1";

    /// <inheritdoc/>
    public string Format => "snac";

    /// <inheritdoc/>
    /// <remarks>Ranks above the generic Hugging Face importer, which also matches a directory with <c>config.json</c> and <c>pytorch_model.bin</c>.</remarks>
    public int Priority => 100;

    /// <inheritdoc/>
    public bool CanOpen(string path)
    {
        if (!Directory.Exists(path) || !File.Exists(Path.Combine(path, "pytorch_model.bin")) || !File.Exists(Path.Combine(path, "config.json")))
        {
            return false;
        }

        try
        {
            using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(path, "config.json")));
            return document.RootElement.ValueKind == JsonValueKind.Object && document.RootElement.TryGetProperty("vq_strides", out _)
                   && document.RootElement.TryGetProperty("decoder_rates", out _);
        }
        catch (JsonException)
        {
            return false;
        }
    }

    /// <inheritdoc/>
    ModelSource IModelImporter.Open(string path) => Open(path);

    /// <summary>Opens a SNAC checkpoint directory.</summary>
    /// <param name="path">The directory.</param>
    /// <exception cref="InvalidDataException">The directory is not a SNAC checkpoint.</exception>
    public static ModelSource Open(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(path, "config.json")));
        var config = document.RootElement.Clone();
        var file = TorchCheckpointFile.Open(Path.Combine(path, "pytorch_model.bin"));
        try
        {
            string a = Architecture + ".";
            var metadata = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                [CanonicalKeys.Architecture] = Architecture,
                [CanonicalKeys.Name] = Path.GetFileName(path.TrimEnd(Path.DirectorySeparatorChar)),
                [a + "sample_rate"] = config.GetProperty("sampling_rate").GetInt32(),
                [a + "decoder_dim"] = config.GetProperty("decoder_dim").GetInt32(),
                [a + "decoder_rates"] = Longs(config, "decoder_rates"),
                [a + "vq_strides"] = Longs(config, "vq_strides"),
                [a + "codebook_size"] = config.GetProperty("codebook_size").GetInt32(),
                [a + "codebook_dim"] = config.GetProperty("codebook_dim").GetInt32(),
                [a + "latent_dim"] = config.TryGetProperty("latent_dim", out var latent) && latent.ValueKind == JsonValueKind.Number
                    ? latent.GetInt32() : config.GetProperty("encoder_dim").GetInt32() * (1 << config.GetProperty("encoder_rates").GetArrayLength()),
                [a + "noise"] = config.GetProperty("noise").GetBoolean(),
                [a + "depthwise"] = config.GetProperty("depthwise").GetBoolean(),
            };
            return new ModelSource
            {
                Format = "snac",
                Architecture = Architecture,
                Metadata = metadata,
                Weights = Weights(file),
                OwnedResources = [file],
                Path = path,
            };
        }
        catch
        {
            file.Dispose();
            throw;
        }
    }

    private static long[] Longs(JsonElement config, string name) => [.. config.GetProperty(name).EnumerateArray().Select(e => e.GetInt64())];

    private static Dictionary<string, WeightInfo> Weights(TorchCheckpointFile file)
    {
        var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        foreach (var (name, entry) in file.Tensors)
        {
            if (name.StartsWith("encoder.", StringComparison.Ordinal))
            {
                continue;
            }

            if (name.EndsWith(Gain, StringComparison.Ordinal))
            {
                string stem = name[..^Gain.Length] + ".weight";
                weights[stem] = TorchWeightNorm.Fold(file, stem, entry, file.Tensors[name[..^Gain.Length] + Direction]);
            }
            else if (!name.EndsWith(Direction, StringComparison.Ordinal))
            {
                bool snake = name.EndsWith(".alpha", StringComparison.Ordinal);
                weights[name] = TorchWeightNorm.View(file, name, entry, snake ? new Shape(entry.Shape[1]) : null);
            }
        }

        return weights;
    }
}
