using System.Runtime.InteropServices;
using Sharplight.Import.HuggingFace;
using Sharplight.Import.PyTorch;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.AceStep;

/// <summary>
/// ACE-Step 1.5 importer: maps the folders of the release (<c>ACE-Step/Ace-Step1.5</c>: a diffusion transformer folder such as <c>acestep-v15-turbo</c> with
/// <c>config.json</c>, <c>model.safetensors</c> and <c>silence_latent.pt</c>; <c>vae</c> with the Oobleck autoencoder; <c>Qwen3-Embedding-0.6B</c>) onto the canonical
/// <see cref="ModelSource"/> of architecture <c>acestep</c> (documented on <c>AceStepAdapter</c>).
/// </summary>
/// <remarks>
/// <para>The checkpoint's <c>decoder.*</c> (the diffusion transformer) becomes <c>dit.*</c> and its <c>encoder.*</c> (the condition encoder) <c>cond.*</c>; the leading axis of the
/// scale-shift tables is dropped and the patch convolutions become f32. The language model planner's audio tokenizer and detokenizer (<c>tokenizer.*</c>,
/// <c>detokenizer.*</c>) are not imported. The text encoder's tensors keep their names under <c>text.</c>, the autoencoder's decoder under <c>vae.decoder.</c> with the weight
/// normalization folded and the snake parameters flattened (its encoder is not imported). The silence latent (<c>silence_latent.pt</c>, a bare tensor stored as
/// <c>[1, 64, frames]</c>) becomes the attachment <c>silence-latent</c>, little-endian f32 <c>[frames, 64]</c>. Only the distilled (turbo) transformer is accepted:
/// the base and SFT variants need classifier-free guidance.</para>
/// <para>Register with <c>services.AddAceStepImporter()</c> so that the <see cref="ModelLoader"/> opens the folders, or call <see cref="Open"/>.</para>
/// </remarks>
public sealed class AceStepImporter : IModelImporter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "acestep";

    private const string VaeFile = "diffusion_pytorch_model.safetensors";

    /// <inheritdoc/>
    public string Format => "acestep";

    /// <inheritdoc/>
    public int Priority => 100;

    /// <inheritdoc/>
    public bool CanOpen(string path) => Directory.Exists(path) && FindTransformer(path) is not null && Directory.Exists(Path.Combine(path, "vae"))
        && FindTextEncoder(path) is not null;

    /// <inheritdoc/>
    ModelSource IModelImporter.Open(string path) => Open(path);

    /// <summary>Opens an ACE-Step 1.5 checkpoint folder.</summary>
    /// <param name="path">The folder with the diffusion transformer, <c>vae</c> and text encoder subfolders.</param>
    /// <exception cref="InvalidDataException">The folder is not an ACE-Step checkpoint.</exception>
    /// <exception cref="NotSupportedException">The transformer is not the distilled variant.</exception>
    public static ModelSource Open(string path)
    {
        string transformer = FindTransformer(path) ?? throw new InvalidDataException($"'{path}' has no ACE-Step diffusion transformer folder.");
        string textDir = FindTextEncoder(path) ?? throw new InvalidDataException($"'{path}' has no Qwen3 text encoder folder.");
        string vaeDir = Path.Combine(path, "vae");
        var owned = new List<IDisposable>();
        try
        {
            using var config = HfConfig.Load(transformer);
            if (config.GetBool("is_turbo") != true)
            {
                throw new NotSupportedException("Only the distilled (turbo) ACE-Step transformer is supported: the base and SFT variants need classifier-free guidance.");
            }

            using var textConfig = HfConfig.Load(textDir);
            using var vaeConfig = HfConfig.Load(vaeDir);
            var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
            var metadata = new Dictionary<string, object>(StringComparer.Ordinal);

            var dit = SafetensorsFile.Open(Path.Combine(transformer, "model.safetensors"));
            owned.Add(dit);
            MapTransformer(dit, weights);
            var text = SafetensorsFile.Open(Path.Combine(textDir, "model.safetensors"));
            owned.Add(text);
            foreach (var (name, tensor) in text.Tensors)
            {
                weights["text." + name] = HfTransforms.Plain(tensor.ToWeightInfo(), "text." + name);
            }

            var vae = SafetensorsFile.Open(Path.Combine(vaeDir, VaeFile));
            owned.Add(vae);
            OobleckDecoderWeights.Map(vae.Tensors, "vae.", w => weights[w.Name] = w);
            HfTokenizerTranslator.AddTo(metadata, textDir, textConfig);

            var attachments = new Dictionary<string, IWeightSource>(StringComparer.Ordinal);
            int silenceFrames = AddSilence(Path.Combine(transformer, "silence_latent.pt"), attachments);
            AddHyperParameters(config, textConfig, vaeConfig, silenceFrames, metadata);
            metadata[CanonicalKeys.Architecture] = Architecture;
            metadata[CanonicalKeys.Name] = Path.GetFileName(path.TrimEnd(Path.DirectorySeparatorChar)) + "/" + Path.GetFileName(transformer);
            return new ModelSource
            {
                Format = "acestep",
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

    private static string? FindTransformer(string path)
    {
        foreach (string dir in Directory.EnumerateDirectories(path, "acestep-v15-*").Order(StringComparer.Ordinal))
        {
            string config = Path.Combine(dir, "config.json");
            if (File.Exists(config) && File.Exists(Path.Combine(dir, "model.safetensors")) && File.Exists(Path.Combine(dir, "silence_latent.pt"))
                && File.ReadAllText(config).Contains("\"acestep\"", StringComparison.Ordinal))
            {
                return dir;
            }
        }

        return null;
    }

    private static string? FindTextEncoder(string path)
        => Directory.EnumerateDirectories(path, "Qwen3-Embedding*").Order(StringComparer.Ordinal)
            .FirstOrDefault(d => File.Exists(Path.Combine(d, "model.safetensors")) && File.Exists(Path.Combine(d, "tokenizer.json")));

    private static void MapTransformer(SafetensorsFile file, Dictionary<string, WeightInfo> weights)
    {
        foreach (var (name, tensor) in file.Tensors)
        {
            string? canonical = name switch
            {
                _ when name.StartsWith("decoder.", StringComparison.Ordinal) => "dit." + name["decoder.".Length..],
                _ when name.StartsWith("encoder.", StringComparison.Ordinal) => "cond." + name["encoder.".Length..],
                "null_condition_emb" => "dit.null_condition_emb",
                _ => null,
            };
            if (canonical is null || tensor.Type is null)
            {
                continue;
            }

            var info = tensor.ToWeightInfo();
            if (canonical.EndsWith("scale_shift_table", StringComparison.Ordinal))
            {
                weights[canonical] = HfTransforms.ToF32(info, canonical, new Shape(info.Shape[1].Fixed, info.Shape[2].Fixed));
            }
            else if (canonical.StartsWith("dit.proj_in.", StringComparison.Ordinal) || canonical.StartsWith("dit.proj_out.", StringComparison.Ordinal))
            {
                weights[canonical] = HfTransforms.ToF32(info, canonical);
            }
            else if (canonical.EndsWith("special_token", StringComparison.Ordinal) || canonical.EndsWith("null_condition_emb", StringComparison.Ordinal))
            {
                weights[canonical] = HfTransforms.ToF32(info, canonical, new Shape(info.Shape[info.Shape.Rank - 1].Fixed));
            }
            else
            {
                weights[canonical] = HfTransforms.Plain(info, canonical);
            }
        }
    }

    // silence_latent.pt is a bare tensor [1, 64, frames] stored as [frames, 64]: the attachment is that storage order (frame-major f32).
    private static int AddSilence(string file, Dictionary<string, IWeightSource> attachments)
    {
        using var checkpoint = TorchCheckpointFile.Open(file, new TorchCheckpointOptions { AllowPermuted = true });
        var entry = checkpoint.Tensors.Values.Single();
        if (entry.DType != TorchDType.Float32 || entry.Shape.Length != 3 || entry.Shape[0] != 1)
        {
            throw new InvalidDataException($"The silence latent must be an f32 tensor [1, channels, frames], not {entry.DType} [{string.Join(", ", entry.Shape)}].");
        }

        int channels = (int)entry.Shape[1];
        int frames = (int)entry.Shape[2];
        var raw = new byte[entry.ByteLength];
        new TorchTensorWeightSource(checkpoint, entry).Read(0, raw);
        var storage = MemoryMarshal.Cast<byte, float>(raw).ToArray();
        float[] logical = entry.Permutation is { } permutation ? TorchPermutation.ToLogical<float>(storage, entry.Shape, permutation) : storage;
        var frameMajor = new float[frames * channels];
        for (int c = 0; c < channels; c++)
        {
            for (int f = 0; f < frames; f++)
            {
                frameMajor[(f * channels) + c] = logical[(c * frames) + f];
            }
        }

        attachments["silence-latent"] = new MemoryWeightSource(MemoryMarshal.AsBytes(frameMajor.AsSpan()).ToArray());
        return frames;
    }

    private static void AddHyperParameters(HfConfig c, HfConfig text, HfConfig vae, int silenceFrames, Dictionary<string, object> md)
    {
        const string A = Architecture + ".";
        var layerTypes = c.GetStringArray("layer_types") ?? throw new InvalidDataException("The transformer's config.json has no layer_types.");
        md[A + "hidden_size"] = c.Int("hidden_size");
        md[A + "intermediate_size"] = c.Int("intermediate_size");
        md[A + "layer_count"] = c.Int("num_hidden_layers");
        md[A + "lyric_layer_count"] = c.Int("num_lyric_encoder_hidden_layers");
        md[A + "timbre_layer_count"] = c.Int("num_timbre_encoder_hidden_layers");
        md[A + "head_count"] = c.Int("num_attention_heads");
        md[A + "kv_head_count"] = c.Int("num_key_value_heads");
        md[A + "head_dim"] = c.Int("head_dim");
        md[A + "norm_eps"] = (float)c.Double("rms_norm_eps");
        md[A + "rope_base"] = (float)c.Double("rope_theta");
        md[A + "sliding_window"] = c.Int("sliding_window");
        md[A + "sliding_layers"] = layerTypes.Select(t => t == "sliding_attention" ? 1L : 0L).ToArray();
        md[A + "patch_size"] = c.Int("patch_size");
        md[A + "latent_channels"] = c.Int("audio_acoustic_hidden_dim");
        md[A + "in_channels"] = c.Int("in_channels");
        md[A + "text.hidden_size"] = text.Int("hidden_size");
        md[A + "text.layer_count"] = text.Int("num_hidden_layers");
        md[A + "text.intermediate_size"] = text.Int("intermediate_size");
        md[A + "text.head_count"] = text.Int("num_attention_heads");
        md[A + "text.kv_head_count"] = text.Int("num_key_value_heads");
        md[A + "text.head_dim"] = text.Int("head_dim");
        md[A + "text.rope_base"] = (float)text.Double("rope_theta");
        md[A + "text.norm_eps"] = (float)text.Double("rms_norm_eps");
        var ratios = vae.Get("downsampling_ratios")!.Value.EnumerateArray().Select(v => v.GetInt64()).ToArray();
        md[A + "sample_rate"] = vae.Int("sampling_rate");
        md[A + "audio_channels"] = vae.Int("audio_channels");
        md[A + "samples_per_frame"] = (int)ratios.Aggregate(1L, (a, b) => a * b);
        md[A + "vae.channels"] = vae.Int("decoder_channels");
        md[A + "vae.channel_multiples"] = vae.Get("channel_multiples")!.Value.EnumerateArray().Select(v => v.GetInt64()).ToArray();
        md[A + "vae.upsampling_ratios"] = ratios.Reverse().ToArray();
        md[A + "silence_frames"] = silenceFrames;
        md[A + "reference_frames"] = c.Int("timbre_fix_frame");
        md[A + "default_steps"] = 8;
        md[A + "default_shift"] = 3f;
        md[A + "chunk_mask_value"] = 2f;
    }
}
