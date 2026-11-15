using System.Text;

namespace Sharplight.Tokenizers.Spm;

/// <summary>
/// SentencePiece-style tokenizer ("spm"; llama.cpp's <c>LLAMA_VOCAB_TYPE_SPM</c>): text is split into UTF-8 characters
/// (spaces become ▁), adjacent pairs whose concatenation is a vocabulary token are merged best-score-first, symbols
/// that never became tokens fall back to byte tokens <c>&lt;0xXX&gt;</c>. Used by Llama 2, Mistral, Gemma, Phi-3.
/// </summary>
internal sealed class SentencePieceTokenizer : ByteFallbackTokenizer
{
    private static readonly string[] EogNames =
    [
        "<|eot_id|>", "<|eot|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>", "</s>", "<eos>", "<|end_of_text|>", "[EOS]",
    ];

    private readonly float[] _scores;
    private readonly Dictionary<string, int> _toId = new(StringComparer.Ordinal);   // Latin-1 view of the UTF-8 bytes

    private struct Symbol
    {
        public int Start;
        public int Len;
        public int Prev;
        public int Next;
    }

    private readonly record struct Bigram(int Left, int Right, float Score, int Size);

    public SentencePieceTokenizer(TokenizerData data)
        : base(data, defaultBosId: 1, defaultEosId: 2, defaultAddSpacePrefix: true)
    {
        int vocab = Tokens.Length;
        _scores = data.Scores is { } scores && scores.Length >= vocab ? scores : new float[vocab];
        for (int i = 0; i < vocab; i++)
        {
            _toId[Latin1(Encoding.UTF8.GetBytes(Tokens[i]))] = i;
        }

        ResolveEndOfGeneration(data, EogNames, name => _toId.TryGetValue(Latin1(Encoding.UTF8.GetBytes(name)), out int id) ? id : null);
    }

    private static string Latin1(ReadOnlySpan<byte> bytes) => Encoding.Latin1.GetString(bytes);

    /// <inheritdoc/>
    public override int? TokenId(string piece) => _toId.TryGetValue(Latin1(Encoding.UTF8.GetBytes(piece)), out int id) ? id : null;

    /// <inheritdoc/>
    public override int[] Encode(string text, bool addSpecial = true, bool parseSpecial = true)
    {
        ArgumentNullException.ThrowIfNull(text);
        var output = new List<int>(Math.Max(16, text.Length / 3 + 4));
        if (addSpecial && AddBos && BosId is int bos)
        {
            output.Add(bos);
        }

        bool previousWasSpecial = true;   // prefix with a space if first
        foreach (var (fragment, id) in Specials.Split(text, parseSpecial))
        {
            if (id >= 0)
            {
                output.Add(id);
                previousWasSpecial = true;
                continue;
            }

            string withPrefix = (AddSpacePrefix && previousWasSpecial ? " " : "") + fragment;
            EncodeText(Encoding.UTF8.GetBytes(withPrefix.Replace(" ", "▁")), output);
            previousWasSpecial = false;
        }

        if (addSpecial && AddEos && EosId is int eos)
        {
            output.Add(eos);
        }

        return [.. output];
    }

    /// <summary>Port of llama.cpp's <c>llm_tokenizer_spm_session::tokenize</c>.</summary>
    private void EncodeText(byte[] text, List<int> output)
    {
        if (text.Length == 0)
        {
            return;
        }

        var symbols = new List<Symbol>(text.Length);
        for (int offset = 0; offset < text.Length;)
        {
            int length = Math.Min(Utf8Length(text[offset]), text.Length - offset);
            int next = offset + length == text.Length ? -1 : symbols.Count + 1;
            symbols.Add(new Symbol { Start = offset, Len = length, Prev = symbols.Count - 1, Next = next });
            offset += length;
        }

        var queue = new PriorityQueue<Bigram, (float, int)>();   // min-heap on (-score, left)
        var mergeSources = new Dictionary<string, (int Left, int Right)>(StringComparer.Ordinal);

        void TryAdd(int left, int right)
        {
            if (left == -1 || right == -1)
            {
                return;
            }

            int size = symbols[left].Len + symbols[right].Len;
            string key = Latin1(text.AsSpan(symbols[left].Start, size));
            if (!_toId.TryGetValue(key, out int id))
            {
                return;
            }

            float score = _scores[id];
            queue.Enqueue(new Bigram(left, right, score, size), (-score, left));
            mergeSources[key] = (left, right);
        }

        for (int i = 1; i < symbols.Count; i++)
        {
            TryAdd(i - 1, i);
        }

        while (queue.Count > 0)
        {
            var bigram = queue.Dequeue();
            var left = symbols[bigram.Left];
            var right = symbols[bigram.Right];
            if (left.Len == 0 || right.Len == 0 || left.Len + right.Len != bigram.Size)
            {
                continue;
            }

            left.Len += right.Len;
            right.Len = 0;
            left.Next = right.Next;
            symbols[bigram.Left] = left;
            symbols[bigram.Right] = right;
            if (right.Next >= 0)
            {
                var following = symbols[right.Next];
                following.Prev = bigram.Left;
                symbols[right.Next] = following;
            }

            TryAdd(left.Prev, bigram.Left);
            TryAdd(bigram.Left, left.Next);
        }

        EmitSymbols(text, symbols, mergeSources, output);
    }

    /// <summary>Emits the final symbols, resegmenting those that are not vocabulary entries along the merges that created them.</summary>
    private void EmitSymbols(byte[] text, List<Symbol> symbols, Dictionary<string, (int Left, int Right)> mergeSources, List<int> output)
    {
        var pending = new Stack<int>();
        for (int i = 0; i != -1; i = symbols[i].Next)
        {
            pending.Push(i);
            while (pending.Count > 0)
            {
                var current = symbols[pending.Pop()];
                string key = Latin1(text.AsSpan(current.Start, current.Len));
                if (_toId.TryGetValue(key, out int token))
                {
                    output.Add(token);
                    continue;
                }

                if (!mergeSources.TryGetValue(key, out var parts))
                {
                    for (int j = 0; j < current.Len; j++)
                    {
                        if (ByteToken[text[current.Start + j]] is var byteToken and >= 0)
                        {
                            output.Add(byteToken);
                        }
                    }

                    continue;
                }

                pending.Push(parts.Right);
                pending.Push(parts.Left);
            }
        }
    }

    private static int Utf8Length(byte lead) => lead < 0xC0 ? 1 : lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : 4;
}
