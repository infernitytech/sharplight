using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Vision;

namespace Sharplight.Llm;

/// <summary>
/// One sequence of autoregressive inference over a compiled decoder graph. Owns the position
/// counter; persistent state (KV cache, recurrent state) lives inside the compiled model.
/// Not thread-safe: use one session per sequence and drive it from one thread at a time. Create with
/// <see cref="LlmModel.CreateSession"/>; most callers use it through a <see cref="Generator"/>.
/// <para>Encoder-decoder models (<see cref="IsEncoderDecoder"/>): the first <see cref="Process(ReadOnlySpan{int})"/> of a context takes
/// the source sequence, runs the encoder over it and starts the decoder with the model's start token; later calls continue the
/// decoder (generated tokens). <see cref="Position"/> counts decoder rows.</para>
/// </summary>
public sealed class InferenceSession : IDisposable
{
    private readonly CompiledModel _compiled;
    private readonly ComputeGraph _graph;
    private readonly Value _tokens;
    private readonly Value? _positions;
    private readonly Value? _outputRows;
    private readonly Value? _embeddings;
    private readonly Value? _embeddingMask;
    private readonly Value? _mediaWeightMask;   // layers with a media weight set (visual experts): which rows use it
    private readonly Value? _samplingParams;
    private readonly Value? _lookahead;         // image rows that attend to their whole image (Gemma 3): rows of the pass each row also sees
    private int[] _lookBuf = [];
    private bool _lookSet;                      // whether this pass's lookahead entries are in _lookBuf
    private readonly int[] _paramBuf = new int[SamplingRow.Width];
    private readonly int[] _sampledToken = new int[1];
    private readonly float[] _sampledLogprob = new float[1];
    private SamplingRow? _sampleRow;
    private readonly int _hidden;
    private float[] _lastHidden = [];
    private int _lastHiddenLength;
    private float[] _embBuf = [];
    private float[] _maskBuf = [];
    private float[] _mediaBuf = [];
    private readonly int _posAxes;
    private readonly int _textPositionAxes;   // leading position axes that carry a text token's position (the rest are 0)
    private readonly int _maxTokens;
    private readonly float[] _logits;
    private readonly RunParameters _run = new();
    private int[] _posBuf = [];
    private readonly CompiledModel? _encoder;
    private readonly Value? _encoderOutputInput;
    private readonly int[] _decoderStart;
    private readonly List<int> _pendingSource = [];
    private float[] _encoded = [];
    private int _encodedRows;
    private readonly Value? _sequenceRows;     // one-slot paged form (window-sized caches, SessionKvLayout): per-row (slot, position, run rows, run)
    private readonly Value? _sequenceRuns;
    private readonly Value? _runEndRows;
    private int[] _rowBuf = [];
    private int[] _runEndRowBuf = [];
    private readonly int[] _runBuf = new int[4];

    /// <summary>Number of logits per step (the model's vocabulary size).</summary>
    public int VocabSize { get; }
    /// <summary>Capacity of the context in rows (prompt plus generated tokens).</summary>
    public int ContextLength { get; }
    /// <summary>
    /// The most rows one forward pass takes: longer prompts run in several passes, and <see cref="LastHidden"/> then holds the last pass's rows only.
    /// </summary>
    public int MaxBatchTokens => _maxTokens;
    /// <summary>Number of rows already in the context (cache positions; image rows count individually).</summary>
    public int Position { get; private set; }
    /// <summary>
    /// Rope position of the next text token. Equals <see cref="Position"/> for text; M-RoPE image spans advance it
    /// by less than their row count.
    /// </summary>
    public int RopePosition { get; private set; }
    /// <summary>
    /// Keep the final hidden states (the graph's <c>hidden</c> output, after the output norms) of the rows of every forward pass in
    /// <see cref="LastHidden"/> (models whose next stage reads them, such as XTTS's waveform decoder). Off by default.
    /// </summary>
    public bool CaptureHidden { get; set; }

    /// <summary>The hidden states of the last forward pass's rows ([rows, width] row-major) when <see cref="CaptureHidden"/> is on, else empty.</summary>
    public ReadOnlySpan<float> LastHidden => _lastHidden.AsSpan(0, _lastHiddenLength);

    /// <summary>Width of the media embedding rows the session's graph takes (0 without a multimodal graph).</summary>
    internal int EmbeddingWidth => SupportsEmbeddings ? _hidden : 0;
    /// <summary>Rope position axes per row (0 when the graph has no positions input).</summary>
    internal int PositionAxes => _posAxes;
    /// <summary>Leading position axes that carry a text row's position.</summary>
    internal int TextPositionAxes => _textPositionAxes;
    /// <summary>Whether the session's graph accepts media embeddings (a <see cref="WellKnown.DecodeMultimodalGraph"/>).</summary>
    public bool SupportsEmbeddings => _embeddings is not null && _embeddingMask is not null;
    /// <summary>
    /// Whether the rows of one image attend to each other (the graph takes <see cref="InputRole.AttentionLookahead"/>, Gemma 3): a multimodal
    /// prompt's passes then end before an image that would not fit whole.
    /// </summary>
    public bool BidirectionalImageRows => _lookahead is not null;
    /// <summary>The compiled graph (placement plan, profiling and device state).</summary>
    public CompiledModel Compiled => _compiled;
    /// <summary>
    /// Whether the session runs an encoder-decoder model: the first <see cref="Process(ReadOnlySpan{int})"/> of a context encodes the
    /// source sequence (see the class remarks).
    /// </summary>
    public bool IsEncoderDecoder => _encoder is not null;
    /// <summary>Longest source sequence an encoder-decoder session can encode (0 for decoder-only models).</summary>
    public int MaxSourceTokens => _encoder is null ? 0 : (int)_encoder.Options.MaxTokens;

    /// <param name="compiled">The compiled decoder graph.</param>
    /// <param name="encoder">The encoder of an encoder-decoder model.</param>
    /// <param name="decoderStartToken">The decoder's first token of an encoder-decoder model.</param>
    /// <param name="textPositionAxes">Leading position axes that carry a text row's position.</param>
    /// <param name="kvLayout">
    /// The window-sized cache layout when <paramref name="compiled"/> is the one-slot paged form of the decode graph (<see cref="SessionKvLayout"/>):
    /// the session fills its sequence tables and fixed block tables; null for the plain graph.
    /// </param>
    /// <param name="contextLength">Rows of the context (default: the compiled context length).</param>
    internal InferenceSession(CompiledModel compiled, CompiledModel? encoder = null, int decoderStartToken = 0, int textPositionAxes = WellKnown.DefaultTextPositionAxes,
        SessionKvLayout? kvLayout = null, int? contextLength = null)
    {
        _encoder = encoder;
        _textPositionAxes = textPositionAxes;
        _decoderStart = [decoderStartToken];
        _compiled = compiled;
        _graph = compiled.Graph;
        _tokens = _graph.Input(InputRole.TokenIds);
        _positions = _graph.TryInput(InputRole.Positions);
        _outputRows = _graph.TryInput(InputRole.OutputRows);
        _embeddings = _graph.TryInput(InputRole.Embeddings);
        _embeddingMask = _graph.TryInput(InputRole.EmbeddingMask);
        _mediaWeightMask = _graph.TryInput(InputRole.MediaWeightMask);
        _samplingParams = _graph.TryInput(InputRole.SamplingParameters);
        _lookahead = _graph.TryInput(InputRole.AttentionLookahead);
        _hidden = _embeddings is null ? 0 : (int)_embeddings.Shape[-1].Fixed;
        _posAxes = _positions is null ? 0 : (int)_positions.Shape[-1].Fixed;
        if (_positions is not null && _positions.Rank == 1)
        {
            _posAxes = 1;
        }

        _maxTokens = (int)compiled.Options.MaxTokens;
        ContextLength = contextLength ?? (int)compiled.Options.ContextLength;
        if (kvLayout is not null)
        {
            _sequenceRows = _graph.Input(InputRole.SequenceRows);
            _sequenceRuns = _graph.Input(InputRole.SequenceRuns);
            _runEndRows = _graph.TryInput(InputRole.SequenceRunEndRows);
            compiled.SetInput<int>(_graph.Input(InputRole.KvBlockTable), kvLayout.BlockTable());
            if (_graph.TryInput(InputRole.KvWindowBlockTable) is { } windowTable)
            {
                compiled.SetInput<int>(windowTable, kvLayout.WindowBlockTable());
            }
        }

        VocabSize = (int)_graph.Outputs["logits"].Shape[-1].Fixed;
        _logits = new float[VocabSize];
        _encoderOutputInput = _graph.TryInput(InputRole.EncoderOutput);
        if ((_encoder is null) != (_encoderOutputInput is null))
        {
            throw new ArgumentException("An encoder-decoder session needs both the encoder and a decoder graph with an encoder-output input.");
        }
    }

    /// <summary>Clears the context (KV cache / recurrent state).</summary>
    public void Reset()
    {
        _compiled.ResetState();
        Position = 0;
        RopePosition = 0;
        _pendingSource.Clear();
        _encodedRows = 0;
    }

    /// <summary>
    /// Feeds tokens (prompt or generated) and returns the logits of the last one.
    /// Long inputs are processed in chunks of the compiled maximum batch size.
    /// The returned array is reused by the next call.
    /// </summary>
    /// <exception cref="ArgumentException"><paramref name="tokens"/> is empty.</exception>
    /// <exception cref="InvalidOperationException">The tokens do not fit the remaining context.</exception>
    public float[] Process(ReadOnlySpan<int> tokens)
    {
        if (tokens.Length == 0)
        {
            throw new ArgumentException("No tokens.");
        }

        if (_encoder is not null && Position == 0)
        {
            // A new context of an encoder-decoder model: the tokens are the source sequence; the decoder starts from its start token.
            Encode(tokens);
            tokens = _decoderStart;
        }

        if (Position + tokens.Length > ContextLength)
        {
            throw new InvalidOperationException($"Context full ({Position} + {tokens.Length} > {ContextLength}).");
        }

        while (tokens.Length > 0)
        {
            int n = Math.Min(tokens.Length, _maxTokens);
            RunChunk(tokens[..n], wantLogits: n == tokens.Length);
            tokens = tokens[n..];
        }
        return _logits;
    }

    /// <summary>
    /// Encoder-decoder sessions: buffers part of the source sequence of a new context; the next <see cref="Process(ReadOnlySpan{int})"/>
    /// encodes the buffered tokens followed by its own (prompts fed in chunks).
    /// </summary>
    /// <exception cref="InvalidOperationException">The session is not encoder-decoder or its decoder has already started.</exception>
    internal void AppendSource(ReadOnlySpan<int> tokens)
    {
        if (_encoder is null || Position != 0)
        {
            throw new InvalidOperationException("Source tokens can only be added to a new context of an encoder-decoder model.");
        }

        _pendingSource.AddRange(tokens);
    }

    /// <summary>Runs the encoder over the buffered source tokens plus <paramref name="tokens"/> and keeps its output for the decoder.</summary>
    private void Encode(ReadOnlySpan<int> tokens)
    {
        var encoder = _encoder!;
        int[] source = [.. _pendingSource, .. tokens];
        _pendingSource.Clear();
        if (source.Length > MaxSourceTokens)
        {
            throw new InvalidOperationException($"Source sequence too long ({source.Length} > {MaxSourceTokens} tokens).");
        }

        var graph = encoder.Graph;
        encoder.SetInput(graph.Input(InputRole.TokenIds), source.AsSpan());
        if (graph.TryInput(InputRole.Positions) is { } positions)
        {
            encoder.SetInput<int>(positions, Enumerable.Range(0, source.Length).ToArray());
        }

        var run = new RunParameters();
        run.Symbols[WellKnown.TokenSymbol] = source.Length;
        encoder.Run(run);
        int hidden = (int)_encoderOutputInput!.Shape[-1].Fixed;
        if (_encoded.Length < source.Length * hidden)
        {
            _encoded = new float[MaxSourceTokens * hidden];
        }

        encoder.GetOutput<float>(WellKnown.EncoderOutput, _encoded.AsSpan(0, source.Length * hidden));
        _encodedRows = source.Length;
    }

    /// <summary>The logits of the last processed row (the buffer returned by <see cref="Process(ReadOnlySpan{int})"/>).</summary>
    public float[] Logits => _logits;

    /// <summary>True when the compiled graph has a device sampling head (see <see cref="ProcessSampled(ReadOnlySpan{int}, SamplingRow)"/>).</summary>
    internal bool SamplesOnDevice => _samplingParams is not null;

    /// <summary>
    /// Like <see cref="Process(ReadOnlySpan{int})"/>, but chooses the next token on the device with <paramref name="sampling"/> and
    /// downloads only it (and its log-probability when asked for) instead of the logits; <see cref="Logits"/> is not updated.
    /// </summary>
    internal (int Token, float Logprob) ProcessSampled(ReadOnlySpan<int> tokens, SamplingRow sampling)
    {
        _sampleRow = sampling;
        try
        {
            Process(tokens);
        }
        finally
        {
            _sampleRow = null;
        }

        return (_sampledToken[0], _sampledLogprob[0]);
    }

    /// <summary>Multimodal form of <see cref="ProcessSampled(ReadOnlySpan{int}, SamplingRow)"/>.</summary>
    internal (int Token, float Logprob) ProcessSampled(MultimodalPrompt prompt, IReadOnlyList<float[]> imageEmbeddings, SamplingRow sampling)
    {
        _sampleRow = sampling;
        try
        {
            Process(prompt, imageEmbeddings);
        }
        finally
        {
            _sampleRow = null;
        }

        return (_sampledToken[0], _sampledLogprob[0]);
    }

    private void RunChunk(ReadOnlySpan<int> tokens, bool wantLogits)
    {
        int T = tokens.Length;
        _compiled.SetInput(_tokens, tokens);
        if (_positions is not null)
        {
            if (_posBuf.Length < T * _posAxes)
            {
                _posBuf = new int[_maxTokens * _posAxes];
            }

            for (int t = 0; t < T; t++)
            {
                for (int a = 0; a < _posAxes; a++)
                {
                    // Text tokens use the same position on the leading axes (M-RoPE: temporal/height/width), extra axes 0.
                    _posBuf[t * _posAxes + a] = a < _textPositionAxes ? RopePosition + t : 0;
                }
            }

            _compiled.SetInput<int>(_positions, _posBuf.AsSpan(0, T * _posAxes));
        }
        if (SupportsEmbeddings)
        {
            // Pure text rows: no media embeddings selected.
            EnsureEmbeddingBuffers();
            _embBuf.AsSpan(0, T * _hidden).Clear();
            _maskBuf.AsSpan(0, T).Clear();
            _compiled.SetInput<float>(_embeddings!, _embBuf.AsSpan(0, T * _hidden));
            _compiled.SetInput<float>(_embeddingMask!, _maskBuf.AsSpan(0, T));
        }

        if (_mediaWeightMask is not null)
        {
            // Text rows use the text weights.
            EnsureEmbeddingBuffers();
            _mediaBuf.AsSpan(0, T).Clear();
            _compiled.SetInput<float>(_mediaWeightMask, _mediaBuf.AsSpan(0, T));
        }
        Execute(T, wantLogits);
        RopePosition += T;
    }

    private void EnsureEmbeddingBuffers()
    {
        if (_embBuf.Length < _maxTokens * _hidden)
        {
            _embBuf = new float[_maxTokens * _hidden];
        }

        if (_maskBuf.Length < _maxTokens)
        {
            _maskBuf = new float[_maxTokens];
        }

        if (_mediaBuf.Length < _maxTokens)
        {
            _mediaBuf = new float[_maxTokens];
        }
    }

    private void Execute(int T, bool wantLogits)
    {
        if (_outputRows is not null)
        {
            _compiled.SetInput<int>(_outputRows, [T - 1]);
        }

        bool sample = wantLogits && _sampleRow is not null;
        if (_samplingParams is not null)
        {
            if (sample)
            {
                _sampleRow!.Value.WriteTo(_paramBuf);
            }
            else
            {
                SamplingRow.WriteSkip(_paramBuf);
            }

            _compiled.SetInput<int>(_samplingParams, _paramBuf);
        }
        else if (sample)
        {
            throw new InvalidOperationException("This session was compiled without a device sampling head.");
        }

        _run.Symbols[WellKnown.TokenSymbol] = T;
        _run.Symbols["R"] = 1;
        if (_lookahead is not null)
        {
            SetLookahead(T);
        }

        if (_sequenceRows is not null)
        {
            SetSequenceTables(T);
        }

        if (_encoderOutputInput is not null)
        {
            _compiled.SetInput<float>(_encoderOutputInput, _encoded.AsSpan(0, _encodedRows * (int)_encoderOutputInput.Shape[-1].Fixed));
            _run.Symbols[WellKnown.EncoderSymbol] = _encodedRows;
        }

        _run.Scalars[WellKnown.PastScalar] = Position;
        _run.SkipOutputs.Clear();
        if (sample || !wantLogits)
        {
            _run.SkipOutputs.Add("logits");
        }

        _compiled.Run(_run);
        Position += T;
        if (sample)
        {
            _compiled.GetOutput<int>(SamplingHead.TokensOutput, _sampledToken);
            _compiled.GetOutput<float>(SamplingHead.LogprobsOutput, _sampledLogprob);
        }
        else if (wantLogits)
        {
            _compiled.GetOutput<float>("logits", _logits);
        }

        _lastHiddenLength = 0;
        if (CaptureHidden && _compiled.Graph.Outputs.TryGetValue("hidden", out var hidden))
        {
            int length = T * (int)hidden.Shape[-1].Fixed;
            if (_lastHidden.Length < length)
            {
                _lastHidden = new float[length];
            }

            _compiled.GetOutput<float>("hidden", _lastHidden.AsSpan(0, length));
            _lastHiddenLength = length;
        }
    }

    /// <summary>
    /// The <see cref="InputRole.AttentionLookahead"/> input: this pass's entries when <see cref="Process(MultimodalPrompt, IReadOnlyList{float[]})"/>
    /// set some (one per row), else a single ignored entry (no row looks ahead).
    /// </summary>
    private void SetLookahead(int T)
    {
        bool any = _lookSet && _lookBuf.AsSpan(0, T).ContainsAnyExcept(0);
        if (!any && _lookBuf.Length == 0)
        {
            _lookBuf = new int[1];
        }

        if (!any)
        {
            _lookBuf[0] = 0;
        }

        _compiled.SetInput<int>(_lookahead!, _lookBuf.AsSpan(0, any ? T : 1));
        _run.Symbols[WellKnown.LookaheadSymbol] = any ? T : 1;
        _lookSet = false;
    }

    /// <summary>
    /// The pass's rows of the image spans of <paramref name="prompt"/> see the rest of their span within the pass
    /// (<see cref="InputRole.AttentionLookahead"/>; rows [<paramref name="start"/>, <paramref name="start"/> + <paramref name="T"/>)).
    /// </summary>
    private void FillLookahead(MultimodalPrompt prompt, int start, int T)
    {
        if (_lookBuf.Length < _maxTokens)
        {
            _lookBuf = new int[_maxTokens];
        }

        int end = start + T;
        for (int row = start; row < end; row++)
        {
            int img = prompt.ImageIndex[row];
            int last = row;
            while (img >= 0 && last + 1 < end && prompt.ImageIndex[last + 1] == img)
            {
                last++;
            }

            _lookBuf[row - start] = last - row;
        }

        _lookSet = true;
    }

    /// <summary>
    /// Rows of the multimodal pass starting at <paramref name="start"/>: up to the compiled maximum, ending before an image span that would
    /// not fit whole when the image's rows attend to each other (a span longer than a pass is split, its parts seeing within themselves).
    /// </summary>
    private int MultimodalPassRows(MultimodalPrompt prompt, int start)
    {
        int end = Math.Min(prompt.Length, start + _maxTokens);
        if (_lookahead is null || end == prompt.Length)
        {
            return end - start;
        }

        int img = prompt.ImageIndex[end - 1];
        if (img < 0 || prompt.ImageIndex[end] != img)
        {
            return end - start;
        }

        int first = end - 1;
        while (first > start && prompt.ImageIndex[first - 1] == img)
        {
            first--;
        }

        return first > start ? first - start : end - start;
    }

    /// <summary>
    /// One-slot paged form: the pass's rows are one run of slot 0 at cache positions [<see cref="Position"/>, <see cref="Position"/> + T)
    /// (<see cref="InputRole.SequenceRows"/>, <see cref="InputRole.SequenceRuns"/>; bidirectional attention sees up to the run's last row).
    /// </summary>
    private void SetSequenceTables(int T)
    {
        if (_rowBuf.Length < T * 4)
        {
            _rowBuf = new int[_maxTokens * 4];
        }

        for (int t = 0; t < T; t++)
        {
            _rowBuf[t * 4] = 0;
            _rowBuf[t * 4 + 1] = Position + t;
            _rowBuf[t * 4 + 2] = T;
            _rowBuf[t * 4 + 3] = 0;
        }

        _compiled.SetInput<int>(_sequenceRows!, _rowBuf.AsSpan(0, T * 4));
        _runBuf[0] = 0;
        _runBuf[1] = 0;
        _runBuf[2] = T;
        _runBuf[3] = Position;
        _compiled.SetInput<int>(_sequenceRuns!, _runBuf);
        if (_runEndRows is not null)
        {
            if (_runEndRowBuf.Length < T * 4)
            {
                _runEndRowBuf = new int[_maxTokens * 4];
            }

            _rowBuf.AsSpan(0, T * 4).CopyTo(_runEndRowBuf);
            for (int t = 0; t < T; t++)
            {
                _runEndRowBuf[t * 4 + 1] = Position + T - 1;
            }

            _compiled.SetInput<int>(_runEndRows, _runEndRowBuf.AsSpan(0, T * 4));
        }

        _run.Symbols[WellKnown.RunSymbol] = 1;
    }

    /// <summary>
    /// Feeds an assembled multimodal prompt (see <see cref="IMultimodalArchitecture"/>): text rows by token id, image rows
    /// from <paramref name="imageEmbeddings"/> (per image, [tokens, hidden] row-major, as produced by <see cref="VisionEncoder"/>).
    /// Requires a session on the <see cref="WellKnown.DecodeMultimodalGraph"/>. Returns the logits of the last row.
    /// </summary>
    public float[] Process(MultimodalPrompt prompt, IReadOnlyList<float[]> imageEmbeddings)
    {
        if (!SupportsEmbeddings)
        {
            throw new InvalidOperationException($"This session's graph takes no embeddings; create it with CreateSession(\"{WellKnown.DecodeMultimodalGraph}\").");
        }

        if (prompt.Length == 0)
        {
            throw new ArgumentException("Empty prompt.");
        }

        if (prompt.PositionAxes != _posAxes)
        {
            throw new ArgumentException($"Prompt has {prompt.PositionAxes} position axes, the model uses {_posAxes}.");
        }

        if (Position + prompt.Length > ContextLength)
        {
            throw new InvalidOperationException($"Context full ({Position} + {prompt.Length} > {ContextLength}).");
        }

        EnsureEmbeddingBuffers();
        if (_posBuf.Length < _maxTokens * _posAxes)
        {
            _posBuf = new int[_maxTokens * _posAxes];
        }

        var toks = new int[_maxTokens];
        for (int start = 0, T; start < prompt.Length; start += T)
        {
            T = MultimodalPassRows(prompt, start);
            for (int t = 0; t < T; t++)
            {
                int row = start + t;
                int img = prompt.ImageIndex[row];
                toks[t] = prompt.Tokens[row];
                var dst = _embBuf.AsSpan(t * _hidden, _hidden);
                if (img < 0)
                {
                    dst.Clear();
                    _maskBuf[t] = 0;
                    _mediaBuf[t] = prompt.MediaWeights?[row] == true ? 1 : 0;
                }
                else
                {
                    imageEmbeddings[img].AsSpan(prompt.ImageRow[row] * _hidden, _hidden).CopyTo(dst);
                    _maskBuf[t] = 1;
                    _mediaBuf[t] = prompt.MediaWeights?[row] ?? true ? 1 : 0;
                }
            }
            _compiled.SetInput(_tokens, toks.AsSpan(0, T));
            prompt.Positions.AsSpan(start * _posAxes, T * _posAxes).CopyTo(_posBuf);
            if (_positions is not null)
            {
                _compiled.SetInput<int>(_positions, _posBuf.AsSpan(0, T * _posAxes));
            }

            _compiled.SetInput<float>(_embeddings!, _embBuf.AsSpan(0, T * _hidden));
            _compiled.SetInput<float>(_embeddingMask!, _maskBuf.AsSpan(0, T));
            if (_mediaWeightMask is not null)
            {
                _compiled.SetInput<float>(_mediaWeightMask, _mediaBuf.AsSpan(0, T));
            }

            if (_lookahead is not null)
            {
                FillLookahead(prompt, start, T);
            }

            Execute(T, wantLogits: start + T == prompt.Length);
        }
        RopePosition = prompt.NextPosition;
        return _logits;
    }

    /// <summary>Releases the device memory of the compiled graph.</summary>
    public void Dispose()
    {
        _compiled.Dispose();
        _encoder?.Dispose();
    }
}
