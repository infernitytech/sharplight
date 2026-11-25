using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Llm.Embeddings;

/// <summary>
/// Options for loading a span scorer: the settings every model kind shares
/// (<see cref="TaskLoadOptions"/>; <see cref="TaskLoadOptions.MaxBatchTokens"/> is the most tokens of all sequences of a batch
/// together and defaults to 8192 here, so longer inputs cannot be processed) plus the packed-batch ones.
/// </summary>
public class PackedModelLoadOptions : TaskLoadOptions
{
    /// <summary>Creates options with the packed-batch defaults (8192 tokens per batch, F16 matrices on GPUs).</summary>
    public PackedModelLoadOptions()
    {
        MaxBatchTokens = 8192;
        HalfPrecisionMatrices = null;
    }

    /// <summary>Most sequences per forward pass.</summary>
    public int MaxBatchSequences { get; set; } = 256;

    /// <summary>Longest input in tokens (longer inputs are truncated or rejected); 0 = the model's limit.</summary>
    public int MaxSequenceLength { get; set; }

    internal int ResolveMaxLength(ModelDefinition def) => ResolveMaxLength(def, MaxSequenceLength, MaxBatchTokens);

    /// <summary>The longest input of a packed-sequence model: the model's limit (or <paramref name="requested"/> when positive), at most <paramref name="maxBatchTokens"/>.</summary>
    internal static int ResolveMaxLength(ModelDefinition def, int requested, int maxBatchTokens)
    {
        int model = def.Property(WellKnown.ContextLength, 512);
        if (def.SourceMetadata.TryGetValue(CanonicalKeys.TokenizerMaxLength, out var ml))
        {
            model = Math.Min(model, Convert.ToInt32(ml, System.Globalization.CultureInfo.InvariantCulture));
        }

        int limit = requested > 0 ? requested : model;
        return Math.Min(limit, maxBatchTokens);
    }
}
