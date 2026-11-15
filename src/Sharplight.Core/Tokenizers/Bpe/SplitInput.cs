namespace Sharplight.Tokenizers.Bpe;

/// <summary>Code points of a text fragment plus the char offset where each starts (length n+1).</summary>
internal readonly ref struct SplitInput(ReadOnlySpan<char> text, ReadOnlySpan<int> cpts, ReadOnlySpan<int> charStart)
{
    public readonly ReadOnlySpan<char> Text = text;
    public readonly ReadOnlySpan<int> Cpts = cpts;
    public readonly ReadOnlySpan<int> CharStart = charStart;
}
