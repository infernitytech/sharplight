using System.Text;
using Sharplight.Tokenizers.Bpe;

namespace Sharplight.Tokenizers.Spm;

/// <summary>
/// Character-level BPE with ▁ for spaces and byte fallback (model "bpe"; Gemma 4): text is split into Unicode characters
/// (spaces become ▁), characters missing from the vocabulary fall back to byte tokens <c>&lt;0xXX&gt;</c>, then the adjacent pair
/// with the lowest merge rank is merged repeatedly (the Hugging Face BPE model with <c>byte_fallback</c>; no pre-tokenization
/// beyond special-token splitting, so a whole prompt is one BPE unit). Merges are <c>"left right"</c> strings of vocabulary tokens.
/// </summary>
internal sealed class RankedBpeTokenizer : ByteFallbackTokenizer
{
    private static readonly string[] EogNames =
    [
        "<|eot_id|>", "<|eot|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<turn|>", "<|endoftext|>", "</s>", "<eos>", "<|end_of_text|>", "[EOS]",
    ];

    private readonly Dictionary<string, int> _toId = new(StringComparer.Ordinal);
    private readonly MergeTable _merges;

    public RankedBpeTokenizer(TokenizerData data)
        : base(data, defaultBosId: null, defaultEosId: null, defaultAddSpacePrefix: false)
    {
        for (int i = 0; i < Tokens.Length; i++)
        {
            _toId[Tokens[i]] = i;
        }

        _merges = BuildMerges(data.Merges ?? []);
        ResolveEndOfGeneration(data, EogNames, name => _toId.TryGetValue(name, out int id) ? id : null);
    }

    private MergeTable BuildMerges(string[] merges)
    {
        var table = new MergeTable(merges.Length);
        for (int rank = 0; rank < merges.Length; rank++)
        {
            string merge = merges[rank];
            int separator = merge.Length > 1 ? merge.IndexOf(' ', 1) : -1;
            if (separator < 0)
            {
                continue;
            }

            string left = merge[..separator];
            string right = merge[(separator + 1)..];
            if (_toId.TryGetValue(left, out int leftId) && _toId.TryGetValue(right, out int rightId) && _toId.TryGetValue(left + right, out int resultId))
            {
                table.TryAdd(leftId, rightId, rank, resultId);
            }
        }

        return table;
    }

    /// <inheritdoc/>
    public override int? TokenId(string piece) => _toId.TryGetValue(piece, out int id) ? id : null;

    /// <inheritdoc/>
    public override int[] Encode(string text, bool addSpecial = true, bool parseSpecial = true)
    {
        ArgumentNullException.ThrowIfNull(text);
        var output = new List<int>(Math.Max(16, text.Length / 3 + 4));
        if (addSpecial && AddBos && BosId is int bos)
        {
            output.Add(bos);
        }

        bool previousWasSpecial = true;
        foreach (var (fragment, id) in Specials.Split(text, parseSpecial))
        {
            if (id >= 0)
            {
                output.Add(id);
                previousWasSpecial = true;
                continue;
            }

            string withPrefix = (AddSpacePrefix && previousWasSpecial ? " " : "") + fragment;
            EncodeText(withPrefix.Replace(' ', '▁'), output);
            previousWasSpecial = false;
        }

        if (addSpecial && AddEos && EosId is int eos)
        {
            output.Add(eos);
        }

        return [.. output];
    }

    /// <summary>Initial symbols: one per character, or one byte token per UTF-8 byte of a character missing from the vocabulary.</summary>
    private int[] InitialSymbols(string text)
    {
        var ids = new List<int>(text.Length);
        for (int i = 0; i < text.Length;)
        {
            int length = char.IsHighSurrogate(text[i]) && i + 1 < text.Length ? 2 : 1;
            string character = text.Substring(i, length);
            if (_toId.TryGetValue(character, out int id))
            {
                ids.Add(id);
            }
            else
            {
                foreach (byte b in Encoding.UTF8.GetBytes(character))
                {
                    if (ByteToken[b] >= 0)
                    {
                        ids.Add(ByteToken[b]);
                    }
                }
            }

            i += length;
        }

        return ids.ToArray();
    }

    private void EncodeText(string text, List<int> output)
    {
        if (text.Length == 0)
        {
            return;
        }

        var symbols = InitialSymbols(text);
        int n = symbols.Length;
        if (n == 0)
        {
            return;
        }

        var previous = new int[n];
        var next = new int[n];
        for (int i = 0; i < n; i++)
        {
            previous[i] = i - 1;
            next[i] = i + 1 < n ? i + 1 : -1;
        }

        var queue = new PriorityQueue<BpeBigram, (int, int)>();

        void TryAdd(int left, int right)
        {
            if (left < 0 || right < 0)
            {
                return;
            }

            if (_merges.TryGet(symbols[left], symbols[right], out int rank, out int result))
            {
                queue.Enqueue(new BpeBigram(left, right, symbols[left], symbols[right], result), (rank, left));
            }
        }

        for (int i = 1; i < n; i++)
        {
            TryAdd(i - 1, i);
        }

        while (queue.Count > 0)
        {
            var bigram = queue.Dequeue();
            bool stale = symbols[bigram.Left] != bigram.LeftId || symbols[bigram.Right] != bigram.RightId
                || symbols[bigram.Left] < 0 || symbols[bigram.Right] < 0 || next[bigram.Left] != bigram.Right;
            if (stale)
            {
                continue;
            }

            symbols[bigram.Left] = bigram.Result;
            symbols[bigram.Right] = -1;
            next[bigram.Left] = next[bigram.Right];
            if (next[bigram.Right] >= 0)
            {
                previous[next[bigram.Right]] = bigram.Left;
            }

            TryAdd(previous[bigram.Left], bigram.Left);
            TryAdd(bigram.Left, next[bigram.Left]);
        }

        for (int i = 0; i != -1; i = next[i])
        {
            output.Add(symbols[i]);
        }
    }
}
