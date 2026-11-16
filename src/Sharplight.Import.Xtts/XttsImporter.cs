using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Audio;
using Sharplight.Import.HuggingFace;
using Sharplight.Import.PyTorch;
using Sharplight.Models;

namespace Sharplight.Import.Xtts;

/// <summary>
/// XTTS v2 importer: maps Coqui's XTTS v2 checkpoint folder (<c>config.json</c> with <c>model_args</c>, <c>model.pth</c>, <c>vocab.json</c>, optionally
/// <c>speakers_xtts.pth</c>) onto the canonical <see cref="ModelSource"/> of architecture <c>xtts</c>.
/// </summary>
/// <remarks>
/// <para>The GPT-2 that generates the audio codes is mapped like a Hugging Face GPT-2 (fused attention, transposed <c>Conv1D</c> weights) to the canonical
/// decoder weights (unprefixed; no position embedding of its own: the learned text and audio positions are added to the input rows); its <c>ln_f</c> is
/// the <c>output_norm</c>, XTTS's <c>final_norm</c> the second output norm <c>output_norm_2</c>, the audio head the <c>lm_head</c> (with its bias), the
/// audio-code embedding the <c>token_embedding</c>. The text embedding and the two learned position tables keep their names under <c>gpt.</c>, the
/// HiFi-GAN waveform decoder under <c>hifigan_decoder.waveform_decoder.</c> (weight norms folded). The tokenizer (<c>vocab.json</c>, a character BPE)
/// becomes canonical tokenizer metadata; every built-in speaker the attachments <c>voices/{name}/latents.f32</c> (the GPT conditioning latents
/// [32, 1024]) and <c>voices/{name}/speaker.f32</c> (the decoder's speaker embedding [512]).</para>
/// <para>Register with <c>services.AddXttsImporter()</c> so that the <see cref="ModelLoader"/> opens XTTS checkpoints, or call <see cref="Open"/>.</para>
/// </remarks>
public sealed partial class XttsImporter : IModelImporter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "xtts";

    private const string Gain = ".parametrizations.weight.original0";
    private const string Direction = ".parametrizations.weight.original1";

    /// <inheritdoc/>
    public string Format => "xtts";

    /// <inheritdoc/>
    public int Priority => 100;

    /// <inheritdoc/>
    public bool CanOpen(string path)
    {
        if (!Directory.Exists(path) || !File.Exists(Path.Combine(path, "model.pth")) || !File.Exists(Path.Combine(path, "vocab.json"))
            || !File.Exists(Path.Combine(path, "config.json")))
        {
            return false;
        }

        try
        {
            using var config = ReadConfig(path);
            return config.RootElement.TryGetProperty("model", out var model) && model.GetString() == "xtts";
        }
        catch (JsonException)
        {
            return false;
        }
    }

    /// <inheritdoc/>
    ModelSource IModelImporter.Open(string path) => Open(path);

    // Coqui writes Python's bare Infinity / NaN (max_audio_len, max_text_len), which JSON does not have: read them as null.
    private static JsonDocument ReadConfig(string path)
        => JsonDocument.Parse(NonFinite().Replace(File.ReadAllText(Path.Combine(path, "config.json")), "${prefix}null"));

    [GeneratedRegex(@"(?<prefix>[:\[,]\s*)-?(Infinity|NaN)\b")]
    private static partial Regex NonFinite();

    /// <summary>Opens an XTTS v2 checkpoint folder.</summary>
    /// <param name="path">The folder.</param>
    /// <exception cref="InvalidDataException">The folder is not an XTTS checkpoint.</exception>
    public static ModelSource Open(string path)
    {
        using var configDocument = ReadConfig(path);
        var args = configDocument.RootElement.GetProperty("model_args");
        int Int(string key, int fallback) => args.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : fallback;
        int layers = Int("gpt_layers", 30);
        int width = Int("gpt_n_model_channels", 1024);
        int heads = Int("gpt_n_heads", 16);
        int audioTokens = Int("gpt_num_audio_tokens", 1026);
        var owned = new List<IDisposable>();
        try
        {
            // The training checkpoint keeps its config object (XttsConfig, a coqpit dataclass) beside the weights: built inert, nothing runs.
            var checkpointOptions = new TorchCheckpointOptions { StateDictKey = "model", InertModulePrefixes = ["TTS.", "coqpit", "trainer."] };
            var file = SafetensorsFile.OpenTorch(Path.Combine(path, "model.pth"), checkpointOptions);
            owned.Add(file);
            var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
            var gpt = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
            foreach (var (name, tensor) in file.Tensors)
            {
                if (tensor.Type is null)
                {
                    continue;
                }

                if (name.StartsWith("gpt.gpt.h.", StringComparison.Ordinal) || name.StartsWith("gpt.gpt.ln_f.", StringComparison.Ordinal))
                {
                    gpt[name["gpt.gpt.".Length..]] = tensor;
                }
                else if (name == "gpt.mel_embedding.weight")
                {
                    gpt["wte.weight"] = tensor;
                }
                else if (name == "gpt.mel_head.weight")
                {
                    gpt["lm_head.weight"] = tensor;
                }
                else if (name == "gpt.mel_head.bias")
                {
                    weights["lm_head.bias"] = tensor.ToWeightInfo().As("lm_head.bias");
                }
                else if (name is "gpt.final_norm.weight" or "gpt.final_norm.bias")
                {
                    string canonical = "output_norm_2." + name[(name.LastIndexOf('.') + 1)..];
                    weights[canonical] = tensor.ToWeightInfo().As(canonical);
                }
                else if ((name.StartsWith("gpt.conditioning_encoder.", StringComparison.Ordinal) || name.StartsWith("gpt.conditioning_perceiver.", StringComparison.Ordinal)
                    || name.StartsWith("hifigan_decoder.speaker_encoder.", StringComparison.Ordinal)) && !name.EndsWith(".num_batches_tracked", StringComparison.Ordinal))
                {
                    // The cloning encoders: the style encoder of the GPT and the decoder's speaker encoder.
                    weights[name] = tensor.ToWeightInfo().As(name);
                }
                else if (name is "gpt.text_embedding.weight" or "gpt.text_pos_embedding.emb.weight" or "gpt.mel_pos_embedding.emb.weight")
                {
                    weights[name] = tensor.ToWeightInfo().As(name);
                }
                else if (name.StartsWith("hifigan_decoder.waveform_decoder.", StringComparison.Ordinal) && !name.EndsWith(Direction, StringComparison.Ordinal))
                {
                    if (name.EndsWith(Gain, StringComparison.Ordinal))
                    {
                        string module = name[..^Gain.Length];
                        weights[module + ".weight"] = HfWeightNormFolding.Fold(tensor.ToWeightInfo(), file.Tensors[module + Direction].ToWeightInfo(), module + ".weight");
                    }
                    else
                    {
                        weights[name] = tensor.ToWeightInfo().As(name);
                    }
                }
            }

            var metadata = new Dictionary<string, object>(StringComparer.Ordinal);
            string gptConfig = $$"""
                { "model_type": "gpt2", "n_embd": {{width}}, "n_layer": {{layers}}, "n_head": {{heads}}, "vocab_size": {{audioTokens}}, "n_positions": 4096,
                  "layer_norm_epsilon": 1e-05, "activation_function": "gelu_new", "tie_word_embeddings": false }
                """;
            using (var config = HfConfig.Parse(gptConfig, path))
            {
                var mapping = new Gpt2HfMapping();
                var options = new HuggingFaceImportOptions();
                foreach (var (key, value) in mapping.MapHyperParameters(config, options))
                {
                    metadata[key] = value;
                }

                foreach (var (name, w) in mapping.MapWeights(config, gpt, options))
                {
                    weights[name] = w;
                }
            }

            using (var tokenizerConfig = HfConfig.Parse("{}", path))
            {
                HfTokenizerTranslator.AddFromFile(metadata, Path.Combine(path, "vocab.json"), path, tokenizerConfig);
            }

            var attachments = new Dictionary<string, IWeightSource>(StringComparer.Ordinal);
            string melStats = Path.Combine(path, "mel_stats.pth");
            if (File.Exists(melStats) && weights.ContainsKey("gpt.conditioning_encoder.init.weight"))
            {
                // torch.save(tensor): a single entry named "tensor".
                var stats = SafetensorsFile.OpenTorch(melStats, new TorchCheckpointOptions { StateDictKey = null });
                owned.Add(stats);
                weights["gpt.mel_stats"] = stats.Tensors["tensor"].ToWeightInfo().As("gpt.mel_stats");
                AddCloningFilters(attachments);
            }

            string speakers = Path.Combine(path, "speakers_xtts.pth");
            if (File.Exists(speakers))
            {
                using var voices = TorchCheckpointFile.Open(speakers, new TorchCheckpointOptions { AllowPermuted = true, StateDictKey = null });
                AddVoices(voices, attachments);
            }

            const string A = Architecture + ".";
            metadata[CanonicalKeys.Architecture] = Architecture;
            metadata[CanonicalKeys.Name] = Path.GetFileName(path.TrimEnd(Path.DirectorySeparatorChar));
            metadata[A + "start_text_token"] = 261;
            metadata[A + "stop_text_token"] = 0;
            metadata[A + "start_audio_token"] = Int("gpt_start_audio_token", 1024);
            metadata[A + "stop_audio_token"] = Int("gpt_stop_audio_token", 1025);
            metadata[A + "max_audio_tokens"] = Int("gpt_max_audio_tokens", 605);
            metadata[A + "code_stride"] = Int("gpt_code_stride_len", 1024);
            metadata[A + "output_hop"] = Int("output_hop_length", 256);
            metadata[A + "input_sample_rate"] = Int("input_sample_rate", 22050);
            metadata[A + "sample_rate"] = Int("output_sample_rate", 24000);
            metadata[A + "speaker_width"] = Int("d_vector_dim", 512);
            // The sampling settings of the model's inference (top level of config.json).
            var root = configDocument.RootElement;
            float Float(string key, float fallback) => root.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetSingle() : fallback;
            metadata[A + "temperature"] = Float("temperature", 0.75f);
            metadata[A + "top_k"] = (int)Float("top_k", 50);
            metadata[A + "top_p"] = Float("top_p", 0.85f);
            metadata[A + "repetition_penalty"] = Float("repetition_penalty", 5f);
            return new ModelSource
            {
                Format = "xtts",
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

    // speakers_xtts.pth: { name: { gpt_cond_latent [1, 32, 1024], speaker_embedding [1, 512, 1] } }.
    private static void AddVoices(TorchCheckpointFile file, Dictionary<string, IWeightSource> attachments)
    {
        foreach (var (name, entry) in file.Tensors)
        {
            int dot = name.LastIndexOf('.');
            if (dot <= 0 || entry.DType != TorchDType.Float32)
            {
                continue;
            }

            string voice = name[..dot];
            string part = name[(dot + 1)..] switch
            {
                "gpt_cond_latent" => "latents.f32",
                "speaker_embedding" => "speaker.f32",
                _ => "",
            };
            if (part.Length == 0)
            {
                continue;
            }

            var bytes = new byte[entry.ByteLength];
            new TorchTensorWeightSource(file, entry).Read(0, bytes);
            if (entry.Permutation is { } permutation)
            {
                var logical = TorchPermutation.ToLogical<float>(MemoryMarshal.Cast<byte, float>(bytes), entry.Shape, permutation);
                bytes = MemoryMarshal.AsBytes(logical.AsSpan()).ToArray();
            }

            attachments[$"voices/{voice}/{part}"] = new MemoryWeightSource(bytes);
        }
    }

    // The mel filter banks of the cloning front ends (torchaudio's, HTK scale): the style encoder's 80 bins up to 8 kHz with Slaney's norm at
    // 22.05 kHz (n_fft 2048), the speaker encoder's 64 bins without norm at 16 kHz (n_fft 512).
    private static void AddCloningFilters(Dictionary<string, IWeightSource> attachments)
    {
        attachments["style-features/mel-filters"] = new MemoryWeightSource(
            () => MemoryMarshal.AsBytes(HtkMelFilterBank.Create(80, 2048, 22050, 0.0, 8000.0, slaneyNorm: true).AsSpan()).ToArray());
        attachments["speaker-features/mel-filters"] = new MemoryWeightSource(
            () => MemoryMarshal.AsBytes(HtkMelFilterBank.Create(64, 512, 16000).AsSpan()).ToArray());
    }
}
