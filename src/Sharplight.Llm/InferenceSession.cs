using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;

namespace Sharplight.Llm;

/// <summary>
/// One sequence of autoregressive inference over a compiled decoder graph. Owns the position
/// counter; persistent state (KV cache, recurrent state) lives inside the compiled model.
/// </summary>
public sealed class InferenceSession : IDisposable
{
    readonly CompiledModel _compiled;
    readonly ComputeGraph _graph;
    readonly Value _tokens;
    readonly Value? _positions;
    readonly Value? _outputRows;
    readonly int _posAxes;
    readonly int _maxTokens;
    readonly float[] _logits;
    readonly RunParameters _run = new();
    int[] _posBuf = [];

    public int VocabSize { get; }
    public int ContextLength { get; }
    /// <summary>Number of tokens already in the context.</summary>
    public int Position { get; private set; }
    public CompiledModel Compiled => _compiled;

    public InferenceSession(ModelDefinition model, CompiledModel compiled)
    {
        _compiled = compiled;
        _graph = compiled.Graph;
        _tokens = _graph.Input(InputRole.TokenIds);
        _positions = _graph.TryInput(InputRole.Positions);
        _outputRows = _graph.TryInput(InputRole.OutputRows);
        _posAxes = _positions is null ? 0 : (int)_positions.Shape[-1].Fixed;
        if (_positions is not null && _positions.Rank == 1) _posAxes = 1;
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
    }

    /// <summary>
    /// Feeds tokens (prompt or generated) and returns the logits of the last one.
    /// Long inputs are processed in chunks of the compiled maximum batch size.
    /// The returned array is reused by the next call.
    /// </summary>
    public float[] Process(ReadOnlySpan<int> tokens)
    {
        if (tokens.Length == 0) throw new ArgumentException("No tokens.");
        if (Position + tokens.Length > ContextLength)
            throw new InvalidOperationException($"Context full ({Position} + {tokens.Length} > {ContextLength}).");
        while (tokens.Length > 0)
        {
            int n = Math.Min(tokens.Length, _maxTokens);
            RunChunk(tokens[..n], wantLogits: n == tokens.Length);
            tokens = tokens[n..];
        }
        return _logits;
    }

    public float[] Logits => _logits;

    void RunChunk(ReadOnlySpan<int> tokens, bool wantLogits)
    {
        int T = tokens.Length;
        _compiled.SetInput(_tokens, tokens);
        if (_positions is not null)
        {
            if (_posBuf.Length < T * _posAxes) _posBuf = new int[_maxTokens * _posAxes];
            for (int t = 0; t < T; t++)
                for (int a = 0; a < _posAxes; a++)
                    // Text tokens use the same position on the temporal/height/width axes (M-RoPE), extra axes 0.
                    _posBuf[t * _posAxes + a] = a < 3 ? Position + t : 0;
            _compiled.SetInput<int>(_positions, _posBuf.AsSpan(0, T * _posAxes));
        }
        if (_outputRows is not null)
            _compiled.SetInput<int>(_outputRows, [T - 1]);

        _run.Symbols[WellKnown.TokenSymbol] = T;
        _run.Symbols["R"] = 1;
        _run.Scalars[WellKnown.PastScalar] = Position;
        _compiled.Run(_run);
        Position += T;
        if (wantLogits) _compiled.GetOutput<float>("logits", _logits);
    }

    public void Dispose() => _compiled.Dispose();
}
