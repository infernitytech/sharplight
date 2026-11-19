using System.Text.Json;
using Sharplight.Import.HuggingFace;
using Sharplight.Models;

namespace Sharplight.Import.StableAudio;

/// <summary>
/// Stable Audio 3 importer: maps the checkpoint folders of the stable-audio-3 release (<c>stabilityai/stable-audio-3-*</c>: <c>model_config.json</c> in the
/// stable-audio-tools layout, <c>model.safetensors</c> with the conditioners, the diffusion transformer and the SAME autoencoder, and the T5Gemma text encoder in a
/// subfolder such as <c>t5gemma-b-b-ul2</c>) onto the canonical <see cref="ModelSource"/> of architecture <c>stable_audio_3</c> (documented on <c>StableAudio3Adapter</c>).
/// </summary>
/// <remarks>
/// <para>Weights: the T5Gemma encoder (only the encoder of the encoder-decoder checkpoint) under <c>text.</c> with the Gemma norms' <c>1 + w</c> folded, the conditioner's
/// learned padding row as <c>text.padding</c>, the seconds conditioner under <c>seconds.</c>, the diffusion transformer under <c>dit.</c> and the autoencoder's decoder
/// under <c>vae.</c> with its weight normalization folded (see <see cref="StableAudioWeights"/>). The autoencoder's encoder (audio to latents, for inpainting and audio
/// variations) is not imported.</para>
/// <para>Register with <c>services.AddStableAudioImporter()</c> so that the <see cref="ModelLoader"/> opens the folders, or call <see cref="Open"/>.</para>
/// </remarks>
public sealed class StableAudioImporter : IModelImporter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "stable_audio_3";

    private const string ConfigFile = "model_config.json";
    private const string WeightsFile = "model.safetensors";

    /// <inheritdoc/>
    public string Format => "stable-audio";

    /// <inheritdoc/>
    public int Priority => 100;

    /// <inheritdoc/>
    public bool CanOpen(string path)
    {
        if (!Directory.Exists(path) || !File.Exists(Path.Combine(path, ConfigFile)) || !File.Exists(Path.Combine(path, WeightsFile)))
        {
            return false;
        }

        try
        {
            using var config = JsonDocument.Parse(File.ReadAllText(Path.Combine(path, ConfigFile)));
            return TextConfig(config.RootElement) is { } text && Directory.Exists(Path.Combine(path, TextFolder(text)));
        }
        catch (JsonException)
        {
            return false;
        }
    }

    /// <inheritdoc/>
    ModelSource IModelImporter.Open(string path) => Open(path);

    /// <summary>Opens a Stable Audio 3 checkpoint folder.</summary>
    /// <param name="path">The folder with <c>model_config.json</c>, <c>model.safetensors</c> and the T5Gemma folder.</param>
    /// <exception cref="InvalidDataException">The folder is not a Stable Audio 3 checkpoint.</exception>
    /// <exception cref="NotSupportedException">The checkpoint uses a part of the stable-audio-3 format that is not supported.</exception>
    public static ModelSource Open(string path)
    {
        using var config = JsonDocument.Parse(File.ReadAllText(Path.Combine(path, ConfigFile)));
        var root = config.RootElement;
        var text = TextConfig(root) ?? throw new InvalidDataException($"'{path}' is not a Stable Audio 3 checkpoint: its conditioning has no t5gemma prompt conditioner.");
        string textDir = Path.Combine(path, TextFolder(text));
        var owned = new List<IDisposable>();
        try
        {
            using var textConfig = HfConfig.Load(textDir);
            var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
            var metadata = new Dictionary<string, object>(StringComparer.Ordinal);
            var file = SafetensorsFile.Open(Path.Combine(path, WeightsFile));
            owned.Add(file);
            var textFile = SafetensorsFile.Open(Path.Combine(textDir, WeightsFile));
            owned.Add(textFile);
            var model = root.GetProperty("model");
            var dit = model.GetProperty("diffusion").GetProperty("config");
            bool differential = dit.TryGetProperty("attn_kwargs", out var attention) && attention.TryGetProperty("differential", out var d) && d.GetBoolean();
            StableAudioWeights.Map(file.Tensors, weights);
            StableAudioWeights.MapTextEncoder(textFile.Tensors, weights);
            HfTokenizerTranslator.AddTo(metadata, textDir, textConfig);
            StableAudioHyperParameters.Add(root, textConfig, differential, metadata);
            metadata[CanonicalKeys.Architecture] = Architecture;
            metadata[CanonicalKeys.Name] = Path.GetFileName(path.TrimEnd(Path.DirectorySeparatorChar));
            return new ModelSource
            {
                Format = "stable-audio",
                Architecture = Architecture,
                Metadata = metadata,
                Weights = weights,
                OwnedResources = owned,
                Path = path,
            };
        }
        catch
        {
            foreach (var resource in owned)
            {
                resource.Dispose();
            }

            throw;
        }
    }

    // The t5gemma conditioner's configuration, or null when the checkpoint has none (not Stable Audio 3).
    private static JsonElement? TextConfig(JsonElement root)
    {
        if (root.ValueKind != JsonValueKind.Object || !root.TryGetProperty("model", out var model) || !model.TryGetProperty("conditioning", out var conditioning)
            || !conditioning.TryGetProperty("configs", out var configs))
        {
            return null;
        }

        foreach (var c in configs.EnumerateArray())
        {
            if (c.TryGetProperty("type", out var type) && type.GetString() == "t5gemma")
            {
                return c.GetProperty("config");
            }
        }

        return null;
    }

    private static string TextFolder(JsonElement? text)
        => text is { } t && t.TryGetProperty("subfolder", out var sub) && sub.GetString() is { Length: > 0 } folder ? folder : "t5gemma-b-b-ul2";
}
