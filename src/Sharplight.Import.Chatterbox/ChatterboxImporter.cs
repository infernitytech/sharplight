using System.Runtime.InteropServices;
using Sharplight.Audio;
using Sharplight.Import.HuggingFace;
using Sharplight.Import.PyTorch;
using Sharplight.Models;

namespace Sharplight.Import.Chatterbox;

/// <summary>
/// Chatterbox importer: maps a Chatterbox checkpoint directory (Resemble AI's English release: <c>t3_cfg.safetensors</c>, <c>s3gen.safetensors</c>,
/// <c>tokenizer.json</c>, optionally <c>ve.safetensors</c> and the built-in voice <c>conds.pt</c>) onto the canonical <see cref="ModelSource"/> of
/// architecture <c>chatterbox</c>.
/// </summary>
/// <remarks>
/// <para>T3, the Llama token language model, is mapped like a Hugging Face Llama checkpoint (its configuration is fixed by the release) to the
/// canonical decoder names and <c>decoder.*</c> keys (unprefixed, as the decoder adapter builds them): the speech-token embedding is its
/// <c>token_embedding</c> and the speech head its <c>lm_head</c>; the text embedding, the learned text and speech positions and the conditioning encoder (speaker projection,
/// perceiver resampler, emotion projection) keep their checkpoint names under <c>t3.</c>. S3Gen's flow (token embedding, upsampling conformer
/// encoder, conditional flow-matching U-Net) and HiFT vocoder keep their names under <c>s3gen.</c> with the weight normalizations folded, the voice
/// encoder its names under <c>ve.</c>. The text tokenizer becomes the canonical tokenizer metadata, and the built-in voice the attachments
/// <c>voices/default/*</c> (little-endian f32 or i32: <c>speaker.f32</c> [256], <c>prompt_tokens.i32</c> [150], <c>flow_tokens.i32</c> [N],
/// <c>flow_mel.f32</c> [2N, 80], <c>flow_speaker.f32</c> [192]).</para>
/// <para>Register with <c>services.AddChatterboxImporter()</c> so that the <see cref="ModelLoader"/> opens Chatterbox checkpoints, or call <see cref="Open"/>.</para>
/// </remarks>
public sealed class ChatterboxImporter : IModelImporter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "chatterbox";

    /// <summary>The prefix of the built-in voice's attachments.</summary>
    public const string DefaultVoicePrefix = "voices/default/";

    private const string Gain = ".parametrizations.weight.original0";
    private const string Direction = ".parametrizations.weight.original1";

    // The release's T3 backbone (chatterbox models/t3/llama_configs.py, Llama_520M): T3 does not ship a config.json.
    private const string T3Config = """
        {
          "model_type": "llama", "vocab_size": 8194, "hidden_size": 1024, "intermediate_size": 4096, "num_hidden_layers": 30,
          "num_attention_heads": 16, "num_key_value_heads": 16, "head_dim": 64, "hidden_act": "silu", "rms_norm_eps": 1e-05,
          "max_position_embeddings": 131072, "rope_theta": 500000.0, "tie_word_embeddings": false,
          "rope_scaling": { "factor": 8.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0, "original_max_position_embeddings": 8192, "rope_type": "llama3" }
        }
        """;

    /// <inheritdoc/>
    public string Format => "chatterbox";

    /// <inheritdoc/>
    public int Priority => 100;

    /// <inheritdoc/>
    public bool CanOpen(string path)
        => Directory.Exists(path) && File.Exists(Path.Combine(path, "t3_cfg.safetensors")) && File.Exists(Path.Combine(path, "s3gen.safetensors"))
           && File.Exists(Path.Combine(path, "tokenizer.json"));

    /// <inheritdoc/>
    ModelSource IModelImporter.Open(string path) => Open(path);

    /// <summary>Opens a Chatterbox checkpoint directory.</summary>
    /// <param name="path">The directory.</param>
    /// <exception cref="InvalidDataException">The directory is not a Chatterbox checkpoint.</exception>
    public static ModelSource Open(string path)
    {
        var owned = new List<IDisposable>();
        try
        {
            var t3 = SafetensorsFile.Open(Path.Combine(path, "t3_cfg.safetensors"));
            owned.Add(t3);
            var s3gen = SafetensorsFile.Open(Path.Combine(path, "s3gen.safetensors"));
            owned.Add(s3gen);
            var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
            var metadata = new Dictionary<string, object>(StringComparer.Ordinal);
            using (var config = HfConfig.Parse(T3Config, path))
            {
                MapT3(t3, config, weights, metadata);
            }

            // The text tokenizer is a plain tokenizer.json; its vocabulary is not padded to T3's speech vocabulary.
            using (var tokenizerConfig = HfConfig.Parse("{}", path))
            {
                HfTokenizerTranslator.AddTo(metadata, path, tokenizerConfig);
            }

            MapFolded(s3gen, "s3gen.", weights);
            string ve = Path.Combine(path, "ve.safetensors");
            if (File.Exists(ve))
            {
                var voiceEncoder = SafetensorsFile.Open(ve);
                owned.Add(voiceEncoder);
                MapFolded(voiceEncoder, "ve.", weights);
            }

            var attachments = new Dictionary<string, IWeightSource>(StringComparer.Ordinal);
            string conds = Path.Combine(path, "conds.pt");
            if (File.Exists(conds))
            {
                using var voice = TorchCheckpointFile.Open(conds, new TorchCheckpointOptions { AllowPermuted = true });
                AddVoice(voice, attachments);
            }

            if (weights.ContainsKey("ve.proj.weight"))
            {
                AddCloningFilters(attachments);
            }

            metadata[CanonicalKeys.Architecture] = Architecture;
            metadata[CanonicalKeys.Name] = Path.GetFileName(path.TrimEnd(Path.DirectorySeparatorChar));
            AddHyperParameters(metadata);
            return new ModelSource
            {
                Format = "chatterbox",
                Architecture = Architecture,
                Metadata = metadata,
                Weights = weights,
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

    // T3's Llama through the Hugging Face decoder mapping (renamed to its layout), everything else of T3 under its own name.
    private static void MapT3(SafetensorsFile file, HfConfig config, Dictionary<string, WeightInfo> weights, Dictionary<string, object> metadata)
    {
        var llama = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        foreach (var (name, tensor) in file.Tensors)
        {
            if (name.StartsWith("tfmr.", StringComparison.Ordinal))
            {
                if (name != "tfmr.embed_tokens.weight")
                {
                    llama["model." + name["tfmr.".Length..]] = tensor;
                }
            }
            else if (name == "speech_emb.weight")
            {
                llama["model.embed_tokens.weight"] = tensor;
            }
            else if (name == "speech_head.weight")
            {
                llama["lm_head.weight"] = tensor;
            }
            else
            {
                var w = tensor.ToWeightInfo();
                weights["t3." + name] = w.As("t3." + name);
            }
        }

        var mapping = new DecoderHfMapping();
        var options = new HuggingFaceImportOptions();
        foreach (var (key, value) in mapping.MapHyperParameters(config, options))
        {
            metadata[key] = value;
        }

        foreach (var (name, w) in mapping.MapWeights(config, llama, options))
        {
            weights[name] = w;
        }
    }

    // Every tensor under the prefix, weight-normalized ones folded into a plain weight.
    private static void MapFolded(SafetensorsFile file, string prefix, Dictionary<string, WeightInfo> weights)
    {
        foreach (var (name, tensor) in file.Tensors)
        {
            if (name.EndsWith(Direction, StringComparison.Ordinal) || tensor.Type is null)
            {
                continue;
            }

            if (name.EndsWith(Gain, StringComparison.Ordinal))
            {
                string module = name[..^Gain.Length];
                string canonical = prefix + module + ".weight";
                var direction = file.Tensors[module + Direction].ToWeightInfo();
                weights[canonical] = HfWeightNormFolding.Fold(tensor.ToWeightInfo(), direction, canonical);
                continue;
            }

            weights[prefix + name] = tensor.ToWeightInfo().As(prefix + name);
        }
    }

    // conds.pt: { t3: { speaker_emb, cond_prompt_speech_tokens, emotion_adv }, gen: { prompt_token, prompt_feat, embedding, ... } }; prompt_feat is a
    // transposed view, read in storage order and put back into row-major [frames, 80].
    private static void AddVoice(TorchCheckpointFile file, Dictionary<string, IWeightSource> attachments)
    {
        void Add(string tensorName, string attachment, bool integer)
        {
            if (!file.Tensors.TryGetValue(tensorName, out var entry))
            {
                throw new InvalidDataException($"conds.pt has no '{tensorName}'.");
            }

            var raw = new byte[entry.ByteLength];
            new TorchTensorWeightSource(file, entry).Read(0, raw);
            int count = (int)entry.Shape.Aggregate(1L, (a, d) => a * d);
            var storage = new double[count];
            for (int i = 0; i < count; i++)
            {
                storage[i] = entry.DType switch
                {
                    TorchDType.Float32 => BitConverter.ToSingle(raw, i * 4),
                    TorchDType.Int64 => BitConverter.ToInt64(raw, i * 8),
                    _ => throw new NotSupportedException($"conds.pt: '{tensorName}' has the unsupported type {entry.DType}."),
                };
            }

            var logical = entry.Permutation is { } permutation ? TorchPermutation.ToLogical<double>(storage, entry.Shape, permutation) : storage;
            byte[] bytes = integer
                ? MemoryMarshal.AsBytes(logical.Select(v => (int)v).ToArray().AsSpan()).ToArray()
                : MemoryMarshal.AsBytes(logical.Select(v => (float)v).ToArray().AsSpan()).ToArray();
            attachments[DefaultVoicePrefix + attachment] = new MemoryWeightSource(bytes);
        }

        Add("t3.speaker_emb", "speaker.f32", integer: false);
        Add("t3.cond_prompt_speech_tokens", "prompt_tokens.i32", integer: true);
        Add("gen.prompt_token", "flow_tokens.i32", integer: true);
        Add("gen.prompt_feat", "flow_mel.f32", integer: false);
        Add("gen.embedding", "flow_speaker.f32", integer: false);
    }

    // The mel filter banks of the cloning front ends (librosa's Slaney banks): the voice encoder's 40 bins and the S3 tokenizer's 128 bins at 16 kHz
    // (n_fft 400), the flow prompt's 80 bins up to 8 kHz at 24 kHz (n_fft 1920).
    private static void AddCloningFilters(Dictionary<string, IWeightSource> attachments)
    {
        static MemoryWeightSource Bank(int bins, int fftSize, int sampleRate, double? maxFrequency)
            => new(() => MemoryMarshal.AsBytes(SlaneyMelFilterBank.Create(bins, fftSize, sampleRate, 0.0, maxFrequency).AsSpan()).ToArray());
        attachments["voice-features/mel-filters"] = Bank(40, 400, 16000, 8000);
        attachments["speech-token-features/mel-filters"] = Bank(128, 400, 16000, null);
        attachments["prompt-features/mel-filters"] = Bank(80, 1920, 24000, 8000);
    }

    // The release's fixed T3 and S3Gen settings (chatterbox models/t3/modules/t3_config.py, s3gen/s3gen.py).
    private static void AddHyperParameters(Dictionary<string, object> md)
    {
        const string A = Architecture + ".";
        md[A + "sample_rate"] = 24000;
        md[A + "t3.start_text_token"] = 255;
        md[A + "t3.stop_text_token"] = 0;
        md[A + "t3.start_speech_token"] = 6561;
        md[A + "t3.stop_speech_token"] = 6562;
        md[A + "t3.speech_vocabulary"] = 6561;
        md[A + "t3.prompt_tokens"] = 150;
        md[A + "t3.perceiver_queries"] = 32;
        md[A + "t3.perceiver_heads"] = 4;
        md[A + "s3gen.mel_bins"] = 80;
        md[A + "s3gen.encoder_heads"] = 8;
        md[A + "s3gen.estimator_heads"] = 8;
        md[A + "s3gen.flow_steps"] = 10;
        md[A + "s3gen.guidance"] = 0.7;
        md[A + "hift.upsample_rates"] = new long[] { 8, 5, 3 };
        md[A + "hift.upsample_kernels"] = new long[] { 16, 11, 7 };
        md[A + "hift.fft_size"] = 16;
        md[A + "hift.hop"] = 4;
        md[A + "hift.harmonics"] = 8;
    }
}
