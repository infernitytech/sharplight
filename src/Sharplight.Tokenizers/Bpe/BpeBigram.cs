namespace Sharplight.Tokenizers.Bpe;

/// <summary>A candidate merge of two adjacent symbols; stale entries are detected by re-checking the symbol ids.</summary>
/// <param name="Left">Position of the left symbol.</param>
/// <param name="Right">Position of the right symbol.</param>
/// <param name="LeftId">Symbol id at <paramref name="Left"/> when the bigram was queued.</param>
/// <param name="RightId">Symbol id at <paramref name="Right"/> when the bigram was queued.</param>
/// <param name="Result">Symbol id of the merged pair.</param>
internal readonly record struct BpeBigram(int Left, int Right, int LeftId, int RightId, int Result);
