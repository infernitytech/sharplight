namespace Sharplight.Llm.Serving;

/// <summary>Media content whose embedding rows were computed already (e.g. by a caller's own encoder): the engine uses them as they are.</summary>
public sealed class EncodedMedia : MediaContent
{
    private readonly PassThroughEncoder _encoder;

    /// <summary>Creates the content from its rows.</summary>
    /// <param name="hash">Hash of the content the rows were computed from (prefix-cache key).</param>
    /// <param name="rows">The rows, [rows, <paramref name="width"/>] row-major.</param>
    /// <param name="width">Width of one row (the decoder's <see cref="IBatchDecoder.EmbeddingWidth"/>).</param>
    /// <exception cref="ArgumentException"><paramref name="rows"/> is not a positive whole number of rows of <paramref name="width"/>.</exception>
    public EncodedMedia(UInt128 hash, float[] rows, int width)
        : base(hash, RowCount(rows, width))
    {
        Embeddings = rows;
        _encoder = new PassThroughEncoder(width);
    }

    /// <summary>The rows, [<see cref="MediaContent.Rows"/>, width] row-major.</summary>
    public float[] Embeddings { get; }

    /// <inheritdoc/>
    public override IMediaEncoder Encoder => _encoder;

    private static int RowCount(float[] rows, int width)
    {
        ArgumentNullException.ThrowIfNull(rows);
        if (width <= 0 || rows.Length == 0 || rows.Length % width != 0)
        {
            throw new ArgumentException($"{rows.Length} values are not whole rows of width {width}.", nameof(rows));
        }

        return rows.Length / width;
    }

    /// <summary>Returns the rows of <see cref="EncodedMedia"/> contents.</summary>
    private sealed class PassThroughEncoder(int width) : IMediaEncoder
    {
        public int EmbeddingWidth => width;

        public IReadOnlyList<float[]> Encode(IReadOnlyList<MediaContent> contents) => [.. contents.Select(c => ((EncodedMedia)c).Embeddings)];
    }
}
