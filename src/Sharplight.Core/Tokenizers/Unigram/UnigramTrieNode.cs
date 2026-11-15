namespace Sharplight.Tokenizers.Unigram;

/// <summary>Node of the piece trie of <see cref="UnigramTokenizer"/>: children by character and the id of the piece ending here (-1 = none).</summary>
internal sealed class UnigramTrieNode
{
    public Dictionary<char, UnigramTrieNode>? Next;
    public int Id = -1;
}
