using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Vision;

namespace Sharplight.Llm;

/// <summary>
/// One sequence of autoregressive inference over a compiled decoder graph. Owns the position
/// counter; persistent state (KV cache, recurrent state) lives inside the compiled model.
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
    private readonly int _hidden;
    private float[] _embBuf = [];
    private float[] _maskBuf = [];
    private readonly int _posAxes;
    private readonly int _maxTokens;
    private readonly float[] _logits;
    private readonly RunParameters _run = new();
    private int[] _posBuf = [];

    public int VocabSize { get; }
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
    public CompiledModel Compiled => _compiled;

    public InferenceSession(ModelDefinition model, CompiledModel compiled)
    {
        _compiled = compiled;
        _graph = compiled.Graph;
        _tokens = _graph.Input(InputRole.TokenIds);
        _positions = _graph.TryInput(InputRole.Positions);
        _outputRows = _graph.TryInput(InputRole.OutputRows);
        _embeddings = _graph.TryInput(InputRole.Embeddings);
        _embeddingMask = _graph.TryInput(InputRole.EmbeddingMask);
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
    }

    /// <summary>Clears the context (KV cache / recurrent state).</summary>
    public void Reset()
    {
        _compiled.ResetState();
        Position = 0;
        RopePosition = 0;
    }

    /// <summary>
    /// Feeds tokens (prompt or generated) and returns the logits of the last one.
    /// Long inputs are processed in chunks of the compiled maximum batch size.
    /// The returned array is reused by the next call.
    /// </summary>
    public float[] Process(ReadOnlySpan<int> tokens)
    {
        if (tokens.Length == 0)
        {
            throw new ArgumentException("No tokens.");
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

    public float[] Logits => _logits;

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

        _run.Symbols[WellKnown.TokenSymbol] = T;
        _run.Symbols["R"] = 1;
        _run.Scalars[WellKnown.PastScalar] = Position;
        _compiled.Run(_run);
        Position += T;
        if (wantLogits)
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
                int row = start + t, img = prompt.ImageIndex[row];
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

    public void Dispose() => _compiled.Dispose();
}
