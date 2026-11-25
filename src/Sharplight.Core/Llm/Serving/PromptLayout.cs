namespace Sharplight.Llm.Serving;

/// <summary>
/// How a sequence's rows differ from plain tokens at their own positions: media embedding rows in the prompt
/// (<see cref="GenerationRequest.PromptMedia"/>), explicit M-RoPE positions of the prompt and the position offset of the rows after it,
/// and media weight flags. The engine builds one per image or audio request and turns every input of the sequence (prompt chunks,
/// recomputation after preemption, decode and verification rows) into the decoder's form with <see cref="Apply"/>; a draft head that
/// mirrors the target's rows (<see cref="Speculative.ISpeculator.SetLayout"/>) does the same for its own inputs.
/// </summary>
public sealed class PromptLayout
{
    private readonly int[]? _positions;
    private readonly bool[]? _mediaWeights;
    private readonly int _axes;
    private readonly int _textAxes;
    private readonly PromptMedia[] _media;
    private readonly float[]?[] _rows;

    /// <summary>Creates the layout of a validated request.</summary>
    /// <param name="request">The request (its prompt, media, positions and media weights).</param>
    /// <param name="axes">The decoder's <see cref="IBatchDecoder.PositionAxes"/>.</param>
    /// <param name="textAxes">Leading position axes that carry a text row's position (the rest are 0).</param>
    internal PromptLayout(GenerationRequest request, int axes, int textAxes)
    {
        PromptLength = request.PromptTokens.Length;
        _positions = request.PromptPositions;
        _mediaWeights = request.PromptMediaWeights;
        _axes = Math.Max(1, axes);
        _textAxes = textAxes;
        _media = [.. request.PromptMedia.OrderBy(m => m.Start)];
        _rows = new float[]?[_media.Length];
        PositionDelta = _positions is null ? 0 : (request.NextPosition ?? PromptLength) - PromptLength;
        Content = MergeContent(request, _media);
    }

    /// <summary>Rows of the prompt.</summary>
    public int PromptLength { get; }

    /// <summary>Rope position minus cache position of every row after the prompt (0 unless M-RoPE image spans compressed the prompt's positions).</summary>
    public int PositionDelta { get; }

    /// <summary>The media of the prompt, by start position.</summary>
    public IReadOnlyList<PromptMedia> Media => _media;

    /// <summary>The prefix-cache keys of the prompt's non-text content: <see cref="GenerationRequest.PromptContent"/> plus one per media content.</summary>
    internal IReadOnlyList<PromptContentKey> Content { get; }

    /// <summary>The encoded rows of media <paramref name="index"/>, or null while not encoded.</summary>
    internal float[]? Rows(int index) => _rows[index];

    /// <summary>Sets the encoded rows of media <paramref name="index"/>.</summary>
    internal void SetRows(int index, float[] rows) => _rows[index] = rows;

    /// <summary>
    /// The first media overlapping context rows [<paramref name="start"/>, <paramref name="start"/> + <paramref name="count"/>) whose rows are not
    /// encoded yet, as an index into <see cref="Media"/>, or -1.
    /// </summary>
    internal int FirstMissing(int start, int count)
    {
        for (int i = 0; i < _media.Length; i++)
        {
            if (_media[i].End > start && _media[i].Start < start + count && _rows[i] is null)
            {
                return i;
            }
        }

        return -1;
    }

    /// <summary>
    /// The input with its rows' positions, embedding rows and media weights. Its rows are the context rows starting at <paramref name="start"/>
    /// (cache positions). Row <c>j</c> takes the embedding of context row <paramref name="embeddingStart"/> + <c>j</c> where that is a media row
    /// (default <paramref name="start"/>; a draft head whose row <c>i</c> embeds the token after <c>i</c> passes <paramref name="start"/> + 1). Media
    /// not encoded yet leave their rows' token embeddings in place.
    /// </summary>
    /// <param name="input">The input (its slot, tokens and logits settings are kept).</param>
    /// <param name="start">Cache position of the input's first row.</param>
    /// <param name="embeddingStart">Context row whose embedding the input's first row takes; null for <paramref name="start"/>.</param>
    public SequenceInput Apply(SequenceInput input, int start, int? embeddingStart = null)
    {
        int n = input.Tokens.Length;
        int from = embeddingStart ?? start;
        List<EmbeddingRun>? runs = null;
        for (int i = 0; i < _media.Length; i++)
        {
            var m = _media[i];
            int first = Math.Max(from, m.Start);
            int end = Math.Min(from + n, m.End);
            if (first >= end || _rows[i] is not { } rows)
            {
                continue;
            }

            int width = rows.Length / m.Content.Rows;
            (runs ??= []).Add(new EmbeddingRun(first - from, rows.AsMemory((first - m.Start) * width, (end - first) * width)));
        }

        var result = input with { Embeddings = runs };
        if (start >= PromptLength)
        {
            return result with { PositionOffset = PositionDelta };
        }

        if (_positions is not null)
        {
            result = result with { Positions = PositionsOf(start, n) };
        }

        if (_mediaWeights is not null)
        {
            result = result with { MediaWeights = MediaWeightsOf(start, n) };
        }

        return result;
    }

    /// <summary>Positions of context rows [start, start + n): the prompt's own, then cache position plus <see cref="PositionDelta"/>.</summary>
    private ReadOnlyMemory<int> PositionsOf(int start, int n)
    {
        int inPrompt = Math.Min(n, PromptLength - start);
        if (inPrompt == n)
        {
            return _positions.AsMemory(start * _axes, n * _axes);
        }

        var positions = new int[n * _axes];
        _positions.AsSpan(start * _axes, inPrompt * _axes).CopyTo(positions);
        for (int j = inPrompt; j < n; j++)
        {
            for (int a = 0; a < _textAxes && a < _axes; a++)
            {
                positions[j * _axes + a] = start + j + PositionDelta;
            }
        }

        return positions;
    }

    /// <summary>Media weight flags of context rows [start, start + n): the prompt's own, then false.</summary>
    private ReadOnlyMemory<bool> MediaWeightsOf(int start, int n)
    {
        int inPrompt = Math.Min(n, PromptLength - start);
        if (inPrompt == n)
        {
            return _mediaWeights.AsMemory(start, n);
        }

        var flags = new bool[n];
        _mediaWeights.AsSpan(start, inPrompt).CopyTo(flags);
        return flags;
    }

    private static PromptContentKey[] MergeContent(GenerationRequest request, PromptMedia[] media)
        => [.. request.PromptContent.Concat(media.Select(m => new PromptContentKey(m.Start, m.Content.Rows, m.Content.Hash))).OrderBy(k => k.Start)];
}
