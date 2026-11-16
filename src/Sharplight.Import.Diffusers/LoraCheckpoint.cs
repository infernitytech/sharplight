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
/// that merges it into the canonical weights of the model it is loaded with. LyCORIS variants (LoHa, LoKr, DoRA, Tucker) are rejected.
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

    private static readonly (string Suffix, bool IsDown)[] s_matrices =
    [
        (".lora_down.weight", true), (".lora_up.weight", false), (".lora_A.weight", true), (".lora_B.weight", false),
        (".lora_A.default.weight", true), (".lora_B.default.weight", false), (".lora.down.weight", true), (".lora.up.weight", false),
    ];

    private static readonly (string Prefix, string Component, bool Kohya)[] s_prefixes =
    [
        ("lora_unet_", Denoiser, true), ("lora_transformer_", Denoiser, true), ("lora_te1_", TextEncoder, true), ("lora_te2_", SecondTextEncoder, true),
        ("lora_te_", TextEncoder, true), ("base_model.model.", Denoiser, false), ("unet.", Denoiser, false), ("transformer.", Denoiser, false),
        ("diffusion_model.", Denoiser, false), ("model.diffusion_model.", Denoiser, false), ("text_encoder_2.", SecondTextEncoder, false),
        ("text_encoder.", TextEncoder, false),
    ];

    /// <summary>Whether the tensor names are a LoRA file.</summary>
    /// <param name="names">The tensor names.</param>
    public static bool IsLora(IReadOnlyCollection<string> names)
        => names.Count > 0 && names.Any(n => s_matrices.Any(m => n.EndsWith(m.Suffix, StringComparison.Ordinal)))
           && names.All(n => n.Contains("lora", StringComparison.Ordinal) || n.EndsWith(".alpha", StringComparison.Ordinal)
               || n.Contains("dora_scale", StringComparison.Ordinal));

    /// <summary>Opens the file as a LoRA source.</summary>
    /// <param name="path">The safetensors file.</param>
    /// <exception cref="NotSupportedException">A LyCORIS variant, or a module without its down or up matrix.</exception>
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

    /// <summary>The modules of a LoRA state dict.</summary>
    /// <param name="tensors">The tensors by name.</param>
    /// <exception cref="NotSupportedException">A LyCORIS variant, or a module without its down or up matrix.</exception>
    public static IReadOnlyList<LoraModule> Modules(IReadOnlyDictionary<string, WeightInfo> tensors)
    {
        if (tensors.Keys.FirstOrDefault(n => n.Contains("dora_scale", StringComparison.Ordinal) || n.Contains("hada_", StringComparison.Ordinal)
                || n.Contains("lokr_", StringComparison.Ordinal) || n.Contains("lora_mid", StringComparison.Ordinal)) is { } variant)
        {
            throw new NotSupportedException($"Not supported yet: LyCORIS / DoRA LoRA variants ('{variant}'); plain LoRA (down / up matrices) is supported.");
        }

        var downs = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        var ups = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        foreach (var (name, weight) in tensors)
        {
            foreach (var (suffix, isDown) in s_matrices)
            {
                if (name.EndsWith(suffix, StringComparison.Ordinal))
                {
                    (isDown ? downs : ups)[name[..^suffix.Length]] = weight;
                    break;
                }
            }
        }

        var modules = new List<LoraModule>();
        foreach (var (key, down) in downs)
        {
            if (!ups.TryGetValue(key, out var up))
            {
                throw new NotSupportedException($"LoRA module '{key}' has no up matrix.");
            }

            float? alpha = tensors.TryGetValue(key + ".alpha", out var alphaTensor) ? Scalar(alphaTensor) : null;
            var (component, name, kohya) = Split(key);
            modules.Add(new LoraModule(component, name, kohya, down, up, alpha));
        }

        return modules;
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
