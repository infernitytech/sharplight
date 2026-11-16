using System.Text;
using Sharplight.Import.PyTorch;
using Sharplight.Models;

namespace Sharplight.Import.Kokoro;

/// <summary>
/// Kokoro importer: maps a Kokoro-82M checkpoint directory (<c>config.json</c>, one <c>kokoro*.pth</c> state dictionary, optionally a <c>voices</c>
/// folder of <c>.pt</c> voice packs) onto the canonical <see cref="ModelSource"/> of architecture <c>kokoro</c> (a StyleTTS 2 style text-to-speech
/// model with an ALBERT text encoder, a prosody predictor and an iSTFTNet decoder). Weight-normalized convolutions are folded (see
/// <see cref="KokoroWeights"/>); the symbol vocabulary becomes the attachment <c>tokenizer/vocab.json</c> and every voice pack the attachment
/// <c>voices/{name}.f32</c> (little-endian f32 <c>[510, 1, 256]</c>: row <c>n - 1</c> is the style of an utterance of <c>n</c> phoneme tokens, the first 128
/// values drive the decoder, the last 128 the prosody predictor).
/// </summary>
/// <remarks>Register with <c>services.AddKokoroImporter()</c> so that the <see cref="ModelLoader"/> opens Kokoro checkpoints, or call <see cref="Open"/>.</remarks>
public sealed class KokoroImporter : IModelImporter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "kokoro";

    /// <summary>The attachment holding the symbol vocabulary.</summary>
    public const string VocabularyAttachment = "tokenizer/vocab.json";

    /// <summary>The prefix of the voice attachments.</summary>
    public const string VoicePrefix = "voices/";

    /// <summary>The prefix of the pronunciation lexicon attachments (the files of the checkpoint's <c>lexicon</c> folder).</summary>
    public const string LexiconPrefix = "lexicon/";

    /// <summary>The suffix of the voice attachments.</summary>
    public const string VoiceSuffix = ".f32";

    private const int SampleRate = 24000;

    /// <inheritdoc/>
    public string Format => "kokoro";

    /// <inheritdoc/>
    /// <remarks>True for a directory (or the <c>.pth</c> file inside one) with a Kokoro <c>config.json</c> and a <c>.pth</c> checkpoint.</remarks>
    public bool CanOpen(string path)
    {
        string? directory = Directory.Exists(path) ? path : File.Exists(path) && path.EndsWith(".pth", StringComparison.OrdinalIgnoreCase) ? Path.GetDirectoryName(path) : null;
        return directory is not null && KokoroConfig.Matches(Path.Combine(directory, "config.json")) && CheckpointOf(path, directory) is not null;
    }

    /// <inheritdoc/>
    ModelSource IModelImporter.Open(string path) => Open(path);

    /// <summary>Opens a Kokoro checkpoint.</summary>
    /// <param name="path">The checkpoint directory, or its <c>.pth</c> file.</param>
    /// <exception cref="InvalidDataException">The directory is not a Kokoro checkpoint.</exception>
    public static ModelSource Open(string path)
    {
        string directory = Directory.Exists(path) ? path : Path.GetDirectoryName(path) ?? ".";
        string checkpoint = CheckpointOf(path, directory) ?? throw new InvalidDataException($"{path}: no Kokoro .pth checkpoint found.");
        var config = KokoroConfig.Read(Path.Combine(directory, "config.json"));
        var owned = new List<IDisposable>();
        try
        {
            var file = TorchCheckpointFile.Open(checkpoint);
            owned.Add(file);
            var attachments = new Dictionary<string, IWeightSource>(StringComparer.Ordinal)
            {
                [VocabularyAttachment] = new MemoryWeightSource(Encoding.UTF8.GetBytes(config.VocabularyJson)),
            };
            AddVoices(Path.Combine(directory, "voices"), attachments, owned);
            AddLexicons(Path.Combine(directory, "lexicon"), attachments);
            var metadata = HyperParameters(config);
            metadata[CanonicalKeys.Architecture] = Architecture;
            metadata[CanonicalKeys.Name] = Path.GetFileName(directory.TrimEnd(Path.DirectorySeparatorChar));
            return new ModelSource
            {
                Format = "kokoro",
                Architecture = Architecture,
                Metadata = metadata,
                Weights = KokoroWeights.Map(file),
                Attachments = attachments,
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

    private static string? CheckpointOf(string path, string directory)
    {
        if (File.Exists(path))
        {
            return path;
        }

        var files = Directory.EnumerateFiles(directory, "*.pth").Order(StringComparer.Ordinal).ToList();
        return files.Count == 0 ? null : files[^1];
    }

    private static void AddVoices(string directory, Dictionary<string, IWeightSource> attachments, List<IDisposable> owned)
    {
        if (!Directory.Exists(directory))
        {
            return;
        }

        foreach (string voice in Directory.EnumerateFiles(directory, "*.pt").Order(StringComparer.Ordinal))
        {
            var file = TorchCheckpointFile.Open(voice);
            owned.Add(file);
            var tensor = file.Tensors.Values.Single();
            if (tensor.DType != TorchDType.Float32 || tensor.Shape.Length != 3 || tensor.Shape[1] != 1 || tensor.Shape[2] != 256)
            {
                throw new InvalidDataException($"{voice}: a Kokoro voice pack is a float32 [N, 1, 256] tensor.");
            }

            attachments[VoicePrefix + Path.GetFileNameWithoutExtension(voice) + VoiceSuffix] = new TorchTensorWeightSource(file, tensor);
        }
    }

    private static void AddLexicons(string directory, Dictionary<string, IWeightSource> attachments)
    {
        if (!Directory.Exists(directory))
        {
            return;
        }

        foreach (string lexicon in Directory.EnumerateFiles(directory, "*.json").Order(StringComparer.Ordinal))
        {
            attachments[LexiconPrefix + Path.GetFileName(lexicon)] = new MemoryWeightSource(File.ReadAllBytes(lexicon));
        }
    }

    private static Dictionary<string, object> HyperParameters(KokoroConfig c)
    {
        string a = Architecture + ".";
        var ints = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "vocab_size"] = c.Int(null, "n_token"),
            [a + "hidden_size"] = c.Int(null, "hidden_dim"),
            [a + "style_dim"] = c.Int(null, "style_dim"),
            [a + "layer_count"] = c.Int(null, "n_layer"),
            [a + "max_duration"] = c.Int(null, "max_dur"),
            [a + "text_encoder.kernel_size"] = c.Int(null, "text_encoder_kernel_size"),
            [a + "bert.hidden_size"] = c.Int("plbert", "hidden_size"),
            [a + "bert.head_count"] = c.Int("plbert", "num_attention_heads"),
            [a + "bert.intermediate_size"] = c.Int("plbert", "intermediate_size"),
            [a + "bert.layer_count"] = c.Int("plbert", "num_hidden_layers"),
            [a + "bert.context_length"] = c.Int("plbert", "max_position_embeddings"),
            [a + "generator.initial_channels"] = c.Int("istftnet", "upsample_initial_channel"),
            [a + "generator.upsample_rates"] = c.Ints("istftnet", "upsample_rates"),
            [a + "generator.upsample_kernel_sizes"] = c.Ints("istftnet", "upsample_kernel_sizes"),
            [a + "generator.resblock_kernel_sizes"] = c.Ints("istftnet", "resblock_kernel_sizes"),
            [a + "generator.resblock_dilations"] = string.Join(';', c.Dilations().Select(r => string.Join(',', r))),
            [a + "generator.n_fft"] = c.Int("istftnet", "gen_istft_n_fft"),
            [a + "generator.hop"] = c.Int("istftnet", "gen_istft_hop_size"),
            [a + "audio.sample_rate"] = SampleRate,
        };
        // The harmonic-plus-noise source of the iSTFTNet generator (constants of the StyleTTS 2 code, not part of config.json).
        ints[a + "source.harmonics"] = 9;
        ints[a + "source.sine_amplitude"] = 0.1f;
        ints[a + "source.noise_std"] = 0.003f;
        ints[a + "source.voiced_threshold"] = 10f;
        return ints;
    }
}
