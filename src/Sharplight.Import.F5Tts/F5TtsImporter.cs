using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using Sharplight.Import.HuggingFace;
using Sharplight.Models;

namespace Sharplight.Import.F5Tts;

/// <summary>
/// F5-TTS importer: maps an F5-TTS checkpoint folder (SWivid's release: <c>model_*.safetensors</c> with the EMA transformer under
/// <c>ema_model.transformer.</c>, and <c>vocab.txt</c>, one character per line) onto the canonical <see cref="ModelSource"/> of architecture
/// <c>f5tts</c>: the transformer's weights under <c>f5.</c> with their checkpoint names, the hyper-parameters read from the weight shapes
/// (<c>f5tts.*</c>), the character vocabulary as the attachment <c>tokenizer/characters.json</c> (a JSON array; a character's index is its id) and
/// the reference audio's log-mel front end (Vocos' features: magnitude STFT, HTK mel bank without normalization, natural log with a 1e-5 floor) as
/// the <c>feature-extractor/mel-filters</c> attachment and <c>f5tts.audio.*</c> keys. The Vocos vocoder is the model's <c>vocoder</c> auxiliary source
/// (<see cref="VocosImporter"/>).
/// </summary>
/// <remarks>Register with <c>services.AddF5TtsImporter()</c> so that the <see cref="ModelLoader"/> opens F5-TTS checkpoints, or call <see cref="Open"/>.</remarks>
public sealed class F5TtsImporter : IModelImporter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "f5tts";

    /// <summary>The attachment holding the character vocabulary.</summary>
    public const string CharactersAttachment = "tokenizer/characters.json";

    private const string Prefix = "ema_model.transformer.";
    private const int SampleRate = 24000;
    private const int FftSize = 1024;
    private const int HopLength = 256;

    /// <inheritdoc/>
    public string Format => "f5tts";

    /// <inheritdoc/>
    public int Priority => 100;

    /// <inheritdoc/>
    public bool CanOpen(string path) => Directory.Exists(path) && File.Exists(Path.Combine(path, "vocab.txt")) && Checkpoint(path) is not null;

    /// <inheritdoc/>
    ModelSource IModelImporter.Open(string path) => Open(path);

    /// <summary>Opens an F5-TTS checkpoint folder.</summary>
    /// <param name="path">The folder.</param>
    /// <exception cref="InvalidDataException">The folder is not an F5-TTS checkpoint.</exception>
    public static ModelSource Open(string path)
    {
        string checkpoint = Checkpoint(path) ?? throw new InvalidDataException($"{path}: no F5-TTS model_*.safetensors found.");
        var file = SafetensorsFile.Open(checkpoint);
        try
        {
            var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
            foreach (var (name, tensor) in file.Tensors)
            {
                if (!name.StartsWith(Prefix, StringComparison.Ordinal) || tensor.Type is null || name.EndsWith("rotary_embed.inv_freq", StringComparison.Ordinal))
                {
                    continue;
                }

                string canonical = "f5." + name[Prefix.Length..];
                weights[canonical] = tensor.ToWeightInfo().As(canonical);
            }

            var characters = File.ReadAllLines(Path.Combine(path, "vocab.txt"), Encoding.UTF8);
            long dim = weights["f5.proj_out.weight"].Shape[1].Fixed;
            int mels = (int)weights["f5.proj_out.weight"].Shape[0].Fixed;
            const string A = Architecture + ".";
            var metadata = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                [CanonicalKeys.Architecture] = Architecture,
                [CanonicalKeys.Name] = Path.GetFileName(path.TrimEnd(Path.DirectorySeparatorChar)),
                [A + "dim"] = (int)dim,
                [A + "depth"] = Count(weights, "f5.transformer_blocks."),
                [A + "heads"] = (int)(dim / 64),
                [A + "text_dim"] = (int)weights["f5.text_embed.text_embed.weight"].Shape[1].Fixed,
                [A + "text_blocks"] = Count(weights, "f5.text_embed.text_blocks."),
                [A + "mel_bins"] = mels,
                [A + "time_channels"] = (int)weights["f5.time_embed.time_mlp.0.weight"].Shape[1].Fixed,
                [A + "audio.sample_rate"] = SampleRate,
                [A + "audio.fft_size"] = FftSize,
                [A + "audio.hop_length"] = HopLength,
                [A + "audio.mel_bins"] = mels,
            };
            var attachments = new Dictionary<string, IWeightSource>(StringComparer.Ordinal)
            {
                [CharactersAttachment] = new MemoryWeightSource(JsonSerializer.SerializeToUtf8Bytes(characters, F5TtsJsonContext.Default.StringArray)),
                ["feature-extractor/mel-filters"] = new MemoryWeightSource(() => MemoryMarshal.AsBytes(HtkMelFilters(mels, FftSize, SampleRate).AsSpan()).ToArray()),
            };
            return new ModelSource
            {
                Format = "f5tts",
                Architecture = Architecture,
                Metadata = metadata,
                Weights = weights,
                Attachments = attachments,
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

    /// <summary>torchaudio's mel filter bank with <c>mel_scale="htk"</c> and no normalization: f32 [mels, n_fft / 2 + 1].</summary>
    /// <param name="mels">Mel bins.</param>
    /// <param name="fftSize">The FFT size.</param>
    /// <param name="sampleRate">The sample rate.</param>
    public static float[] HtkMelFilters(int mels, int fftSize, int sampleRate)
    {
        static double ToMel(double hz) => 2595.0 * Math.Log10(1.0 + (hz / 700.0));
        int bins = (fftSize / 2) + 1;
        double top = ToMel(sampleRate / 2.0);
        var points = new double[mels + 2];
        for (int i = 0; i < points.Length; i++)
        {
            double mel = top * i / (mels + 1);
            points[i] = 700.0 * (Math.Pow(10, mel / 2595.0) - 1.0);
        }

        var filters = new float[mels * bins];
        for (int m = 0; m < mels; m++)
        {
            for (int k = 0; k < bins; k++)
            {
                double frequency = (double)k * (sampleRate / 2) / (bins - 1);
                double down = (frequency - points[m]) / (points[m + 1] - points[m]);
                double up = (points[m + 2] - frequency) / (points[m + 2] - points[m + 1]);
                filters[(m * bins) + k] = (float)Math.Max(0, Math.Min(down, up));
            }
        }

        return filters;
    }

    private static string? Checkpoint(string path)
        => Directory.Exists(path) ? Directory.GetFiles(path, "model_*.safetensors").Order(StringComparer.Ordinal).LastOrDefault() : null;

    private static int Count(Dictionary<string, WeightInfo> weights, string prefix)
    {
        int count = 0;
        while (weights.Keys.Any(k => k.StartsWith($"{prefix}{count}.", StringComparison.Ordinal)))
        {
            count++;
        }

        return count;
    }
}
