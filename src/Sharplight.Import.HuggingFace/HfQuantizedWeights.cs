namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Turns the quantized weights of a checkpoint into self-contained block-quantized weights before the architecture mapping sees
/// them: FP8 (<see cref="HfFp8Weights"/>), packed FP4 (<see cref="HfFp4Weights"/>), packed integer codes with group scales
/// (<see cref="HfIntegerWeights"/>: GPTQ, AWQ, compressed-tensors), bitsandbytes (<see cref="HfBitsAndBytesWeights"/>) and MLX
/// affine quantization (<see cref="HfMlxWeights"/>).
/// Each format is recognized by its tensors, so checkpoints that mix formats per module work. Calibration tensors the runtime does
/// not use (static activation and KV-cache scales) are dropped.
/// </summary>
internal static class HfQuantizedWeights
{
    /// <summary>Calibration tensors of quantized checkpoints that the runtime does not use (activation and KV-cache scales).</summary>
    private static readonly string[] s_ignoredSuffixes =
        [".input_scale", ".input_global_scale", ".input_zero_point", ".k_scale", ".v_scale", ".q_scale", ".prob_scale", ".output_scale"];

    /// <summary>The checkpoint tensors with every quantized weight converted and its companion tensors removed.</summary>
    /// <param name="config">The checkpoint config (its <c>quantization_config</c>).</param>
    /// <param name="tensors">The checkpoint tensors.</param>
    public static IReadOnlyDictionary<string, SafetensorsTensor> Resolve(HfConfig config, IReadOnlyDictionary<string, SafetensorsTensor> tensors)
    {
        var resolved = HfFp8Weights.Resolve(config, HfTorchAoLayouts.RenameFloat8(HfQuantoLayouts.RenameFloat8(tensors)));
        resolved = HfQuarkFloatWeights.Resolve(config, resolved);
        resolved = HfFp4Weights.Resolve(resolved);
        resolved = HfIntegerWeights.Resolve(config, resolved);
        resolved = HfBitsAndBytesWeights.Resolve(resolved);
        resolved = HfMlxWeights.Resolve(config, resolved);
        resolved = HfAqlmWeights.Resolve(resolved);
        if (ReferenceEquals(resolved, tensors) && !config.TryGetQuantization("quantization_config", out _))
        {
            return tensors;
        }

        return resolved.Where(pair => !s_ignoredSuffixes.Any(suffix => pair.Key.EndsWith(suffix, StringComparison.Ordinal)))
            .ToDictionary(StringComparer.Ordinal);
    }
}
