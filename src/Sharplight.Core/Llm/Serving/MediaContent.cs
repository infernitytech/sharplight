namespace Sharplight.Llm.Serving;

/// <summary>
/// Non-text prompt content whose embedding rows replace placeholder tokens (<see cref="PromptMedia"/>): an image or an audio clip, with the
/// encoder that computes the rows. Its <see cref="Hash"/> names the content in the prefix cache (the KV blocks it overlaps) and in the engine's
/// cache of encoder outputs, so a repeated image is neither encoded nor prefilled again.
/// </summary>
public abstract class MediaContent
{
    /// <summary>Creates the content.</summary>
    /// <param name="hash">Collision-resistant hash of everything the rows depend on (e.g. 128 bits of SHA-256 of the preprocessed pixels).</param>
    /// <param name="rows">Embedding rows the content fills.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="rows"/> is not positive.</exception>
    protected MediaContent(UInt128 hash, int rows)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(rows, 1);
        Hash = hash;
        Rows = rows;
    }

    /// <summary>Hash of the content (prefix-cache and encoder-cache key).</summary>
    public UInt128 Hash { get; }

    /// <summary>Number of embedding rows (prompt positions) the content fills.</summary>
    public int Rows { get; }

    /// <summary>The encoder that computes the rows; contents with the same encoder are encoded together.</summary>
    public abstract IMediaEncoder Encoder { get; }
}
