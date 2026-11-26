using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Turns the quantized weights of a checkpoint into self-contained block-quantized weights before the architecture mapping sees
/// them: FP8 (<see cref="HfFp8Weights"/>), packed FP4 (<see cref="HfFp4Weights"/>), packed integer codes with group scales
/// (<see cref="HfIntegerWeights"/>: GPTQ, AWQ, compressed-tensors), ggml blocks (<see cref="HfGgmlPackedWeights"/>, ModelOpt mixed precision), bitsandbytes (<see cref="HfBitsAndBytesWeights"/>) and MLX
/// affine quantization (<see cref="HfMlxWeights"/>).
/// Each format is recognized by its tensors, so checkpoints that mix formats per module work. Calibration tensors the runtime does
/// not use (static activation and KV-cache scales) are dropped. Every converted weight carries the activation scheme its module's
/// quantization config declares (<see cref="HfActivationSchemes"/>, <see cref="WeightInfo.Activations"/>), so kernels keep the
/// activations of weight-only modules at full precision.
/// </summary>
internal static class HfQuantizedWeights
{
    /// <summary>Calibration tensors of quantized checkpoints that the runtime does not use (activation and KV-cache scales).</summary>
    private static readonly string[] s_ignoredSuffixes =
        [".input_scale", ".input_global_scale", ".input_zero_point", ".k_scale", ".v_scale", ".q_scale", ".prob_scale", ".output_scale"];

    /// <summary>The checkpoint tensors with every quantized weight converted and its companion tensors removed.</summary>
    /// <param name="config">The checkpoint config (its <c>quantization_config</c>).</param>
    /// <param name="tensors">The checkpoint tensors.</param>
    /// <param name="modelOpt">The <c>quantization</c> object of <c>hf_quant_config.json</c>, when the checkpoint has one.</param>
    public static IReadOnlyDictionary<string, SafetensorsTensor> Resolve(HfConfig config, IReadOnlyDictionary<string, SafetensorsTensor> tensors,
        JsonElement? modelOpt = null)
    {
        var renamed = HfSuffixedScaleLayouts.Rename(HfTorchAoLayouts.RenameFloat8(HfQuantoLayouts.RenameFloat8(tensors)));
        var resolved = HfFp8Weights.Resolve(config, renamed);
        resolved = HfGgmlPackedWeights.Resolve(config, resolved);
        resolved = HfQuarkFloatWeights.Resolve(config, resolved);
        resolved = HfFp4Weights.Resolve(resolved);
        resolved = HfIntegerWeights.Resolve(config, resolved);
        resolved = HfBitsAndBytesWeights.Resolve(resolved);
        resolved = HfMlxWeights.Resolve(config, resolved);
        resolved = HfAqlmWeights.Resolve(resolved);
        resolved = HfFpQuantWeights.Resolve(config, resolved);
        resolved = HfHiggsWeights.Resolve(config, resolved);
        resolved = HfVptqWeights.Resolve(config, resolved);
        resolved = HfExl2Weights.Resolve(resolved);
        if (ReferenceEquals(resolved, tensors) && !config.TryGetQuantization("quantization_config", out _))
        {
            return tensors;
        }

        var schemes = HfActivationSchemes.From(config, modelOpt);
        return resolved.Where(pair => !s_ignoredSuffixes.Any(suffix => pair.Key.EndsWith(suffix, StringComparison.Ordinal)))
            .ToDictionary(pair => pair.Key, pair => WithScheme(pair.Value, schemes), StringComparer.Ordinal);
    }

    /// <summary><paramref name="tensor"/>, a converted weight with the activation scheme its module declares.</summary>
    private static SafetensorsTensor WithScheme(SafetensorsTensor tensor, HfActivationSchemes schemes)
    {
        if (tensor.Converted is not { Activations: ActivationScheme.Unspecified } w || schemes.For(tensor.Name) is var scheme && scheme == ActivationScheme.Unspecified)
        {
            return tensor;
        }

        var converted = new WeightInfo
        {
            Name = w.Name, Type = w.Type, Shape = w.Shape, Source = w.Source, InputOrder = w.InputOrder, ScaleRowBlock = w.ScaleRowBlock, Activations = scheme,
        };
        return new SafetensorsTensor
        {
            Name = tensor.Name, DType = tensor.DType, Type = tensor.Type, Shape = tensor.Shape, Offset = tensor.Offset, ByteLength = tensor.ByteLength,
            File = tensor.File, Converted = converted,
        };
    }
}
