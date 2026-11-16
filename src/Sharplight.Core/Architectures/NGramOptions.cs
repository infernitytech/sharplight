using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Configuration of the hashed n-gram embedding block (see <see cref="NGramEmbedding"/>), read from the canonical
/// <c>{arch}.ngram_embedding.*</c> hyper-parameters.
/// </summary>
/// <param name="Layer">The layer (0-based) whose input receives the n-gram features (<c>ngram_embedding.layer</c>).</param>
/// <param name="Multipliers">Per window position j (0 = the token itself) the odd hash multiplier (<c>ngram_embedding.multipliers</c>); the length is the largest n-gram order.</param>
/// <param name="HeadSizes">Rows per hash head (<c>ngram_embedding.head_sizes</c>), heads of order 2 first.</param>
/// <param name="HeadOffsets">First table row of every hash head (<c>ngram_embedding.head_offsets</c>).</param>
/// <param name="HeadsPerOrder">Hash heads per n-gram order (<c>ngram_embedding.heads_per_order</c>).</param>
/// <param name="StopToken">Token that ends an n-gram context; missing predecessors read as it (<c>ngram_embedding.stop_token</c>).</param>
/// <param name="ConvDilation">Dilation of the depthwise short convolution (<c>ngram_embedding.conv_dilation</c>).</param>
internal sealed record NGramOptions(int Layer, long[] Multipliers, long[] HeadSizes, long[] HeadOffsets, int HeadsPerOrder, long StopToken, int ConvDilation)
{
    /// <summary>Largest n-gram order (window length including the token itself).</summary>
    public int Order => Multipliers.Length;

    /// <summary>Reads the <c>ngram_embedding.*</c> keys of <paramref name="source"/>, or null when the model has none.</summary>
    /// <exception cref="InvalidDataException">The head tables are inconsistent.</exception>
    public static NGramOptions? Read(ModelSource source)
    {
        if (!source.Has($"{source.Architecture}.ngram_embedding.layer"))
        {
            return null;
        }

        var options = new NGramOptions(
            source.Arch<int>("ngram_embedding.layer"),
            source.Arch<long[]>("ngram_embedding.multipliers"),
            source.Arch<long[]>("ngram_embedding.head_sizes"),
            source.Arch<long[]>("ngram_embedding.head_offsets"),
            source.Arch<int>("ngram_embedding.heads_per_order"),
            source.Arch<long>("ngram_embedding.stop_token"),
            source.Arch("ngram_embedding.conv_dilation", 1));
        if (options.Order < 2 || options.HeadSizes.Length != (options.Order - 1) * options.HeadsPerOrder
            || options.HeadOffsets.Length != options.HeadSizes.Length)
        {
            throw new InvalidDataException($"{source.Architecture}: inconsistent ngram_embedding head tables.");
        }

        return options;
    }
}
