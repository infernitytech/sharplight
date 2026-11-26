using System.Runtime.InteropServices;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// A LoRA file for an image generator (<c>.safetensors</c>): kohya-ss names (<c>lora_unet_*</c>, <c>lora_te_*</c> / <c>lora_te1_*</c> /
/// <c>lora_te2_*</c>, <c>lora_transformer_*</c>; <c>.lora_down</c> / <c>.lora_up</c> / <c>.alpha</c>), diffusers / PEFT names
/// (<c>unet.</c> / <c>transformer.</c> / <c>text_encoder.</c> / <c>text_encoder_2.</c> modules with <c>.lora_A</c> / <c>.lora_B</c>, or the
/// older <c>.lora.down</c> / <c>.lora.up</c>) and ComfyUI's (<c>diffusion_model.</c> modules). It opens as a source of architecture
/// <see cref="Architecture"/> whose <see cref="ModelSource.FormatData"/> is the <see cref="IWeightPatch"/> (<see cref="DiffusionLoraPatch"/>)
/// that merges it into the canonical weights of the model it is loaded with. LyCORIS' variants are read too: LoCon (convolution LoRAs, with
/// the optional Tucker core <c>lora_mid</c>), LoHa (<c>hada_w1_a</c> ... <c>hada_t2</c>), LoKr (<c>lokr_w1</c> ... <c>lokr_t2</c>), full
/// differences (<c>diff</c>) and DoRA's magnitudes (<c>dora_scale</c>, PEFT's <c>lora_magnitude_vector</c>).
/// </summary>
internal static class LoraCheckpoint
{
    /// <summary>The architecture id of LoRA sources.</summary>
    public const string Architecture = "lora";

    /// <summary>The denoiser component (UNet or transformer).</summary>
    public const string Denoiser = "denoiser";

    /// <summary>The (first) text encoder.</summary>
    public const string TextEncoder = "text_encoder";

    /// <summary>The second text encoder (SDXL's OpenCLIP bigG).</summary>
    public const string SecondTextEncoder = "text_encoder_2";

    // Module tensors by suffix (the longest match wins): plain LoRA (kohya, PEFT, diffusers' old names), LyCORIS' LoCon Tucker core, LoHa,
    // LoKr and full differences, DoRA magnitudes (LyCORIS / kohya's dora_scale, PEFT's lora_magnitude_vector) and the alpha.
    private static readonly (string Suffix, string Part)[] s_parts = [.. new (string Suffix, string Part)[]
    {
        (".lora_down.weight", "down"), (".lora_up.weight", "up"), (".lora_A.weight", "down"), (".lora_B.weight", "up"),
        (".lora_A.default.weight", "down"), (".lora_B.default.weight", "up"), (".lora.down.weight", "down"), (".lora.up.weight", "up"),
        (".lora_mid.weight", "mid"),
        (".hada_w1_a", "hada_w1_a"), (".hada_w1_b", "hada_w1_b"), (".hada_w2_a", "hada_w2_a"), (".hada_w2_b", "hada_w2_b"),
        (".hada_t1", "hada_t1"), (".hada_t2", "hada_t2"),
        (".lokr_w1", "lokr_w1"), (".lokr_w1_a", "lokr_w1_a"), (".lokr_w1_b", "lokr_w1_b"), (".lokr_w2", "lokr_w2"), (".lokr_w2_a", "lokr_w2_a"),
        (".lokr_w2_b", "lokr_w2_b"), (".lokr_t2", "lokr_t2"),
        (".diff", "diff"),
        (".dora_scale", "magnitude"), (".lora_magnitude_vector", "magnitude"), (".lora_magnitude_vector.weight", "magnitude"),
        (".lora_magnitude_vector.default.weight", "magnitude"),
        (".alpha", "alpha"),
    }.OrderByDescending(p => p.Suffix.Length)];

    // Parts that start a module (a file with none of them is not a LoRA).
    private static readonly string[] s_leadingParts = ["down", "hada_w1_a", "lokr_w1", "lokr_w1_a"];

    private static readonly (string Prefix, string Component, bool Kohya)[] s_prefixes =
    [
        ("lora_unet_", Denoiser, true), ("lora_transformer_", Denoiser, true), ("lora_te1_", TextEncoder, true), ("lora_te2_", SecondTextEncoder, true),
        ("lora_te_", TextEncoder, true), ("base_model.model.", Denoiser, false), ("unet.", Denoiser, false), ("transformer.", Denoiser, false),
        ("diffusion_model.", Denoiser, false), ("model.diffusion_model.", Denoiser, false), ("text_encoder_2.", SecondTextEncoder, false),
        ("text_encoder.", TextEncoder, false),
    ];

    /// <summary>Whether the tensor names are a LoRA-family file (LoRA, LoCon, LoHa, LoKr, DoRA).</summary>
    /// <param name="names">The tensor names.</param>
    public static bool IsLora(IReadOnlyCollection<string> names)
        => names.Count > 0 && names.Any(n => Part(n) is { } part && s_leadingParts.Contains(part.Part))
           && names.All(n => n.Contains("lora", StringComparison.Ordinal) || Part(n) is not null);

    /// <summary>The module key and part of a tensor name, or null for other tensors.</summary>
    private static (string Key, string Part)? Part(string name)
    {
        foreach (var (suffix, part) in s_parts)
        {
            if (name.EndsWith(suffix, StringComparison.Ordinal))
            {
                return (name[..^suffix.Length], part);
            }
        }

        return null;
    }

    /// <summary>Opens the file as a LoRA source.</summary>
    /// <param name="path">The safetensors file.</param>
    /// <exception cref="NotSupportedException">A module without its matrices.</exception>
    public static ModelSource Open(string path)
    {
        var tensors = new CheckpointTensors();
        try
        {
            tensors.Add(path);
            var modules = Modules(tensors.Tensors);
            var metadata = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                [CanonicalKeys.Name] = Path.GetFileNameWithoutExtension(path),
                [CanonicalKeys.Architecture] = Architecture,
            };
            return new ModelSource
            {
                Format = "safetensors",
                Architecture = Architecture,
                Metadata = metadata,
                Weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal),
                OwnedResources = tensors.Resources,
                FormatData = new DiffusionLoraPatch(Path.GetFileName(path), modules),
                Path = path,
            };
        }
        catch
        {
            tensors.DisposeAll();
            throw;
        }
    }

    /// <summary>The modules of a LoRA-family state dict.</summary>
    /// <param name="tensors">The tensors by name.</param>
    /// <exception cref="NotSupportedException">A module without its matrices, or with an incomplete LoHa / LoKr decomposition.</exception>
    public static IReadOnlyList<LoraModule> Modules(IReadOnlyDictionary<string, WeightInfo> tensors)
    {
        var parts = new Dictionary<string, Dictionary<string, WeightInfo>>(StringComparer.Ordinal);
        foreach (var (name, weight) in tensors)
        {
            if (Part(name) is not { } found)
            {
                continue;
            }

            if (!parts.TryGetValue(found.Key, out var module))
            {
                parts[found.Key] = module = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
            }

            module[found.Part] = weight;
        }

        var modules = new List<LoraModule>();
        foreach (var (key, module) in parts)
        {
            float? alpha = module.TryGetValue("alpha", out var alphaTensor) ? Scalar(alphaTensor) : null;
            var (component, name, kohya) = Split(key);
            modules.Add(new LoraModule(component, name, kohya, Delta(key, module), alpha, module.GetValueOrDefault("magnitude")));
        }

        return modules;
    }

    private static LoraDelta Delta(string key, Dictionary<string, WeightInfo> module)
    {
        WeightInfo? Get(string part) => module.GetValueOrDefault(part);
        WeightInfo Need(string part, string kind) => Get(part) ?? throw new NotSupportedException($"{kind} module '{key}' has no '{part}' tensor.");
        if (Get("hada_w1_a") is not null)
        {
            return new HadamardDelta(Need("hada_w1_a", "LoHa"), Need("hada_w1_b", "LoHa"), Need("hada_w2_a", "LoHa"), Need("hada_w2_b", "LoHa"),
                Get("hada_t1"), Get("hada_t2"));
        }

        if (Get("lokr_w1") is not null || Get("lokr_w1_a") is not null)
        {
            var w1 = Get("lokr_w1");
            var w2 = Get("lokr_w2");
            return new KroneckerDelta(w1, w1 is null ? Need("lokr_w1_a", "LoKr") : null, w1 is null ? Need("lokr_w1_b", "LoKr") : null,
                w2, w2 is null ? Need("lokr_w2_a", "LoKr") : null, w2 is null ? Need("lokr_w2_b", "LoKr") : null, Get("lokr_t2"));
        }

        if (Get("down") is { } down)
        {
            return new LowRankDelta(down, Get("up") ?? throw new NotSupportedException($"LoRA module '{key}' has no up matrix."), Get("mid"));
        }

        return Get("diff") is { } difference
            ? new FullDelta(difference)
            : throw new NotSupportedException($"LoRA module '{key}' has no weight difference (only {string.Join(", ", module.Keys)}).");
    }

    private static (string Component, string Name, bool Kohya) Split(string key)
    {
        foreach (var (prefix, component, kohya) in s_prefixes)
        {
            if (key.StartsWith(prefix, StringComparison.Ordinal))
            {
                string rest = key[prefix.Length..];
                // PEFT wraps the module in the model it was trained on (base_model.model.<module>), diffusers' old format adds
                // processor names (attn1.processor.to_q_lora); both reduce to the module name.
                rest = rest.Replace(".processor.", ".", StringComparison.Ordinal);
                foreach (string part in new[] { "to_q", "to_k", "to_v", "to_out" })
                {
                    if (rest.EndsWith("." + part + "_lora", StringComparison.Ordinal))
                    {
                        rest = rest[..^"_lora".Length] + (part == "to_out" ? ".0" : "");
                    }
                }

                return (component, rest, kohya);
            }
        }

        return (Denoiser, key, false);
    }

    private static float Scalar(WeightInfo weight)
    {
        byte[] bytes = weight.ReadAll();
        if (weight.Type == DataType.F32)
        {
            return MemoryMarshal.Read<float>(bytes);
        }

        if (weight.Type == DataType.F16)
        {
            return (float)MemoryMarshal.Read<Half>(bytes);
        }

        if (weight.Type == DataType.BF16)
        {
            return BitConverter.Int32BitsToSingle(MemoryMarshal.Read<ushort>(bytes) << 16);
        }

        return weight.Type == DataType.F64
            ? (float)MemoryMarshal.Read<double>(bytes)
            : throw new NotSupportedException($"{weight.Name}: alpha of type {weight.Type}.");
    }
}
