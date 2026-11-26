using System.Runtime.InteropServices;
using Sharplight.Import.Gguf;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Merges a LoRA into the canonical weights of an image generator: every module is mapped to the canonical weight it adapts (through the
/// same name mappings the importer uses for the model's own tensors: SD 1.x / 2.x / XL UNets and CLIP text encoders from diffusers or kohya
/// names, SDXL's SGM names included; FLUX / Chroma transformers from diffusers names (q / k / v and the single blocks' MLP input as row
/// blocks of the fused projections) or the original names; Qwen-Image's diffusers names), and the weight becomes W + s · α / r · ΔW, where ΔW
/// is up · down for a LoRA / LoCon and the LyCORIS reconstructions for LoHa / LoKr (<see cref="LoraDelta"/>); DoRA modules renormalize the
/// result to their magnitude. Computed when the weight is read (quantized weights are decoded; the result is f16, Q8_0 for quantized weights
/// whose rows allow it, or f32 for f32 weights).
/// </summary>
internal sealed class DiffusionLoraPatch : IWeightPatch
{
    private static readonly WeightInfo s_probe = WeightInfo.Derived("probe", DataType.F32, new Shape(1, 1), () => new byte[sizeof(float)]);

    private readonly string _name;
    private readonly IReadOnlyList<LoraModule> _modules;

    /// <summary>Creates the patch.</summary>
    /// <param name="name">The LoRA's file name (for messages).</param>
    /// <param name="modules">Its modules.</param>
    public DiffusionLoraPatch(string name, IReadOnlyList<LoraModule> modules)
    {
        _name = name;
        _modules = modules;
    }

    /// <summary>The modules.</summary>
    public IReadOnlyList<LoraModule> Modules => _modules;

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> Apply(ModelSource primary, float scale)
    {
        ArgumentNullException.ThrowIfNull(primary);
        var targets = new Dictionary<string, List<(LoraModule Module, long Row, bool Swap)>>(StringComparer.Ordinal);
        foreach (var module in _modules)
        {
            var (canonical, row, swap) = Resolve(primary, module);
            if (canonical is null || !primary.Weights.ContainsKey(canonical))
            {
                throw new NotSupportedException($"{_name}: the LoRA module '{module.Component}.{module.Name}' matches no weight of "
                    + $"{primary.Get(CanonicalKeys.Name, primary.Architecture)} ({primary.Architecture}); is the LoRA for another model family?");
            }

            if (!targets.TryGetValue(canonical, out var list))
            {
                targets[canonical] = list = [];
            }

            list.Add((module, row, swap));
        }

        var result = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        foreach (var (canonical, deltas) in targets)
        {
            result[canonical] = Merge(primary.Weights[canonical], deltas, scale);
        }

        return result;
    }

    /// <summary>The canonical weight a module adapts, the first row its delta goes to and whether its rows' halves are swapped.</summary>
    private static (string? Canonical, long Row, bool Swap) Resolve(ModelSource primary, LoraModule module)
    {
        string arch = primary.Architecture;
        if (module.Component != LoraCheckpoint.Denoiser)
        {
            string name = module.Kohya ? KohyaNames.Dotted(module.Name) : module.Name;
            string prefix = module.Component == LoraCheckpoint.SecondTextEncoder ? LatentDiffusionSource.SecondTextEncoderPrefix : LatentDiffusionSource.TextEncoderPrefix;
            return (Probe(weights => DiffusersNames.AddClipText(name + ".weight", s_probe, prefix, weights)), 0, false);
        }

        switch (arch)
        {
            case LatentDiffusionSource.Architecture:
            {
                int layersPerBlock = primary.Get(LatentDiffusionSource.Architecture + ".unet.layers_per_block", 2);
                string name = module.Kohya ? KohyaNames.UNet(module.Name, layersPerBlock) : module.Name;
                return (Probe(weights => DiffusersNames.AddUNet(name + ".weight", s_probe, weights)), 0, false);
            }

            case FluxConfig.Architecture or ChromaConfig.Architecture:
            {
                string name = (module.Kohya ? KohyaNames.Dotted(module.Name) : module.Name) + ".weight";
                if (IsOriginalFluxName(name))
                {
                    return (FluxNames.OriginalName(name) is { } original ? "transformer." + original : null, 0, false);
                }

                var (canonical, part) = FluxNames.DiffusersName(name);
                if (canonical is null)
                {
                    return (null, 0, false);
                }

                long hidden = primary.Weights["transformer.img_in.weight"].Shape[0].Fixed;
                return ("transformer." + canonical, part < 0 ? 0 : part * hidden, canonical.StartsWith("final.mod.", StringComparison.Ordinal));
            }

            case QwenImageConfig.Architecture:
            {
                string? canonical = QwenImageNames.CanonicalName((module.Kohya ? KohyaNames.Dotted(module.Name) : module.Name) + ".weight");
                return (canonical is null ? null : "transformer." + canonical, 0, canonical?.StartsWith("final.mod.", StringComparison.Ordinal) == true);
            }

            case Krea2Config.Architecture:
            {
                // diffusers / PEFT names (Krea's and ComfyUI's LoRAs) or the original Krea names.
                string name = (module.Kohya ? KohyaNames.Dotted(module.Name) : module.Name) + ".weight";
                return (Probe(weights => Krea2Names.Add(new Dictionary<string, WeightInfo> { [name] = s_probe }, "", weights)), 0, false);
            }

            default:
                throw new NotSupportedException($"Not supported yet: LoRAs for {arch} models (SD 1.x / 2.x / XL, FLUX, Chroma, Qwen-Image and Krea 2 are supported).");
        }
    }

    /// <summary>Whether a FLUX module name is in the original (Black Forest Labs) layout rather than diffusers'.</summary>
    private static bool IsOriginalFluxName(string name)
        => name.StartsWith("double_blocks.", StringComparison.Ordinal) || name.StartsWith("single_blocks.", StringComparison.Ordinal)
           || name.StartsWith("img_in.", StringComparison.Ordinal) || name.StartsWith("txt_in.", StringComparison.Ordinal)
           || name.StartsWith("time_in.", StringComparison.Ordinal) || name.StartsWith("vector_in.", StringComparison.Ordinal)
           || name.StartsWith("guidance_in.", StringComparison.Ordinal) || name.StartsWith("final_layer.", StringComparison.Ordinal);

    /// <summary>The canonical name a mapping function gives a tensor name, or null when it rejects it.</summary>
    private static string? Probe(Action<CanonicalWeights> add)
    {
        var weights = new CanonicalWeights();
        try
        {
            add(weights);
        }
        catch (InvalidDataException)
        {
            return null;
        }

        return weights.Weights.Keys.SingleOrDefault();
    }

    private static WeightInfo Merge(WeightInfo target, List<(LoraModule Module, long Row, bool Swap)> deltas, float scale)
    {
        if (target.InputOrder is not null)
        {
            throw new NotSupportedException($"{target.Name}: a LoRA cannot be merged into a weight with a permuted input order.");
        }

        long rows = target.Shape[0].Fixed;
        long columns = target.Shape.Elements() / rows;
        foreach (var (module, row, _) in deltas)
        {
            var delta = module.Delta;
            if (delta.Columns != columns || row + delta.Rows > rows)
            {
                throw new NotSupportedException($"{target.Name}: the {delta.Kind} module '{module.Name}' ({delta.Rows} x {delta.Columns}) does not fit the weight {target.Shape}.");
            }

            if (module.Magnitude is { } magnitude && magnitude.Shape.Elements() != delta.Rows
                && (magnitude.Shape.Elements() != columns || delta.Rows != rows))
            {
                throw new NotSupportedException($"{target.Name}: the DoRA magnitude {magnitude.Shape} of '{module.Name}' fits neither its rows nor its columns.");
            }
        }

        // A quantized weight stays quantized (Q8_0) where its rows allow it: a LoRA on every linear of a Q8_0 Krea 2 would otherwise double
        // the transformer to f16 (26 GB).
        var type = target.Type == DataType.F32 ? DataType.F32
            : target.Type.IsQuantized && columns % Q8Blocks.BlockSize == 0 ? DataType.Q8_0
            : DataType.F16;
        return WeightInfo.Derived(target.Name, type, target.Shape, () =>
        {
            float[] w = Decode(target);
            foreach (var (module, first, swap) in deltas)
            {
                Apply(w, columns, module, first, swap, scale);
            }

            if (type == DataType.F32)
            {
                return MemoryMarshal.AsBytes(w.AsSpan()).ToArray();
            }

            if (type == DataType.Q8_0)
            {
                return Q8Blocks.Encode(w);
            }

            var halves = new Half[w.Length];
            for (int i = 0; i < w.Length; i++)
            {
                halves[i] = (Half)w[i];
            }

            return MemoryMarshal.AsBytes(halves.AsSpan()).ToArray();
        });
    }

    /// <summary>
    /// Adds one module's scaled difference to the rows from <paramref name="first"/> of <paramref name="w"/>; for DoRA the patched rows are then
    /// renormalized to the module's magnitude (PEFT's merge: m · (W + s · ΔW) / ‖W + s · ΔW‖, the norm per output row, or per input column for
    /// LyCORIS files that store the magnitude per input).
    /// </summary>
    private static void Apply(float[] w, long columns, LoraModule module, long first, bool swap, float scale)
    {
        float[] delta = module.Delta.Compute();
        int outRows = (int)module.Delta.Rows;
        float factor = scale * module.Factor;
        int half = outRows / 2;
        Parallel.For(0, outRows, i =>
        {
            int r = swap ? (i + half) % outRows : i;
            var destination = w.AsSpan((int)((first + r) * columns), (int)columns);
            var source = delta.AsSpan((int)(i * columns), (int)columns);
            for (int c = 0; c < destination.Length; c++)
            {
                destination[c] += factor * source[c];
            }
        });
        if (module.Magnitude is not { } magnitudeWeight)
        {
            return;
        }

        float[] magnitude = Decode(magnitudeWeight);
        if (magnitude.Length == outRows)
        {
            Parallel.For(0, outRows, i =>
            {
                int r = swap ? (i + half) % outRows : i;
                var row = w.AsSpan((int)((first + r) * columns), (int)columns);
                double squares = 0;
                foreach (float value in row)
                {
                    squares += value * (double)value;
                }

                float ratio = squares > 0 ? (float)(magnitude[i] / Math.Sqrt(squares)) : 1f;
                for (int c = 0; c < row.Length; c++)
                {
                    row[c] *= ratio;
                }
            });
            return;
        }

        var norms = new double[columns];
        for (long i = 0; i < outRows; i++)
        {
            for (long c = 0; c < columns; c++)
            {
                double value = w[(first + i) * columns + c];
                norms[c] += value * value;
            }
        }

        for (long i = 0; i < outRows; i++)
        {
            for (long c = 0; c < columns; c++)
            {
                w[(first + i) * columns + c] *= norms[c] > 0 ? (float)(magnitude[c] / Math.Sqrt(norms[c])) : 1f;
            }
        }
    }

    /// <summary>The values of a stored tensor as f32 (f32, f16, bf16 or any ggml type).</summary>
    /// <param name="weight">The tensor.</param>
    internal static float[] Decode(WeightInfo weight)
    {
        byte[] bytes = weight.ReadAll();
        if (weight.Type == DataType.F32)
        {
            return MemoryMarshal.Cast<byte, float>(bytes).ToArray();
        }

        if (weight.Type == DataType.F16 || weight.Type == DataType.BF16)
        {
            var halves = MemoryMarshal.Cast<byte, ushort>(bytes);
            var values = new float[halves.Length];
            bool f16 = weight.Type == DataType.F16;
            for (int i = 0; i < halves.Length; i++)
            {
                values[i] = f16 ? (float)BitConverter.UInt16BitsToHalf(halves[i]) : BitConverter.Int32BitsToSingle(halves[i] << 16);
            }

            return values;
        }

        return GgmlBlockDecoder.Decode(weight.Type, bytes);
    }
}
