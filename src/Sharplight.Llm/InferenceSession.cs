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
    private readonly Value? _samplingParams;
    private readonly int[] _paramBuf = new int[SamplingRow.Width];
    private readonly int[] _sampledToken = new int[1];
    private readonly float[] _sampledLogprob = new float[1];
    private SamplingRow? _sampleRow;
    private readonly int _hidden;
    private float[] _embBuf = [];
    private float[] _maskBuf = [];
    private readonly int _posAxes;
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

    /// <summary>Number of logits per step (the model's vocabulary size).</summary>
    public int VocabSize { get; }
    /// <summary>Capacity of the context in rows (prompt plus generated tokens).</summary>
    public int ContextLength { get; }
    /// <summary>Number of rows already in the context (cache positions; image rows count individually).</summary>
    public int Position { get; private set; }
    /// <summary>
    /// Rope position of the next text token. Equals <see cref="Position"/> for text; M-RoPE image spans advance it
    /// by less than their row count.
    /// </summary>
    public int RopePosition { get; private set; }
    /// <summary>Whether the session's graph accepts media embeddings (a <see cref="WellKnown.DecodeMultimodalGraph"/>).</summary>
    public bool SupportsEmbeddings => _embeddings is not null && _embeddingMask is not null;
    /// <summary>The compiled graph (placement plan, profiling and device state).</summary>
    public CompiledModel Compiled => _compiled;
    /// <summary>
    /// Whether the session runs an encoder-decoder model: the first <see cref="Process(ReadOnlySpan{int})"/> of a context encodes the
    /// source sequence (see the class remarks).
    /// </summary>
    public bool IsEncoderDecoder => _encoder is not null;
    /// <summary>Longest source sequence an encoder-decoder session can encode (0 for decoder-only models).</summary>
    public int MaxSourceTokens => _encoder is null ? 0 : (int)_encoder.Options.MaxTokens;

    internal InferenceSession(CompiledModel compiled, CompiledModel? encoder = null, int decoderStartToken = 0)
    {
        _encoder = encoder;
        _decoderStart = [decoderStartToken];
        _compiled = compiled;
        _graph = compiled.Graph;
        _tokens = _graph.Input(InputRole.TokenIds);
        _positions = _graph.TryInput(InputRole.Positions);
        _outputRows = _graph.TryInput(InputRole.OutputRows);
        _embeddings = _graph.TryInput(InputRole.Embeddings);
        _embeddingMask = _graph.TryInput(InputRole.EmbeddingMask);
        _samplingParams = _graph.TryInput(InputRole.SamplingParameters);
        _hidden = _embeddings is null ? 0 : (int)_embeddings.Shape[-1].Fixed;
        _posAxes = _positions is null ? 0 : (int)_positions.Shape[-1].Fixed;
        if (_positions is not null && _positions.Rank == 1)
        {
            _posAxes = 1;
        }

        _maxTokens = (int)compiled.Options.MaxTokens;
        ContextLength = (int)compiled.Options.ContextLength;
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
                    // Text tokens use the same position on the temporal/height/width axes (M-RoPE), extra axes 0.
                    _posBuf[t * _posAxes + a] = a < 3 ? RopePosition + t : 0;
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
        for (int start = 0; start < prompt.Length; start += _maxTokens)
        {
            int T = Math.Min(_maxTokens, prompt.Length - start);
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
                }
                else
                {
                    imageEmbeddings[img].AsSpan(prompt.ImageRow[row] * _hidden, _hidden).CopyTo(dst);
                    _maskBuf[t] = 1;
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
