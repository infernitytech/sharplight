namespace Sharplight.Tokenizers.Bpe;

/// <summary>Splits a fragment into "words" (lengths measured in code points, summing to the fragment's length).</summary>
internal abstract class PreTokenizer
{
    public abstract void Split(in SplitInput input, List<int> lengths);
}
