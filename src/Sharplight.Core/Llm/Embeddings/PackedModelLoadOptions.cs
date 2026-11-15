using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Llm.Embeddings;

/// <summary>
/// Options for loading an embedding model, a reranker or a span scorer: the settings every model kind shares
/// (<see cref="TaskLoadOptions"/>; <see cref="TaskLoadOptions.MaxBatchTokens"/> is the most tokens of all sequences of a batch
/// together and defaults to 8192 here, so longer inputs cannot be processed) plus the packed-batch ones.
/// </summary>
public class PackedModelLoadOptions : TaskLoadOptions
{
    /// <summary>Creates options with the packed-batch defaults (8192 tokens per batch).</summary>
    public PackedModelLoadOptions()
    {
        MaxBatchTokens = 8192;
    }

    /// <summary>Most sequences per forward pass.</summary>
    public int MaxBatchSequences { get; set; } = 256;

    /// <summary>Longest input in tokens (longer inputs are truncated or rejected); 0 = the model's limit.</summary>
    public int MaxSequenceLength { get; set; }

    /// <summary>
    /// Store F32 matrices (rank ≥ 2 weights, e.g. of Hugging Face checkpoints saved in float32) as F16 at load: half the memory and
    /// f16 matrix units on GPUs, at a cosine similarity to the F32 result of ~0.99999. Null (default) = on when the first device is
    /// a GPU; norms, biases and quantized / 16-bit weights are never changed.
    /// </summary>
    public bool? HalfPrecisionMatrices { get; set; }

    internal int ResolveMaxLength(ModelDefinition def)
    {
        int model = def.Property(WellKnown.ContextLength, 512);
        if (def.SourceMetadata.TryGetValue(CanonicalKeys.TokenizerMaxLength, out var ml))
        {
            model = Math.Min(model, Convert.ToInt32(ml, System.Globalization.CultureInfo.InvariantCulture));
        }

        int limit = MaxSequenceLength > 0 ? MaxSequenceLength : model;
        return Math.Min(limit, MaxBatchTokens);
    }
}
