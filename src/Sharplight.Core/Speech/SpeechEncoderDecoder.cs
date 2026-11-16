using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.Speech;

/// <summary>
/// Runs the compiled graphs of an encoder-decoder speech model: the encoder over one window of audio features (<see cref="SpeechEncoder"/>),
/// and the decoder token by token with its KV cache. The first decoder pass of a window carries the encoder output (the decoder caches its
/// cross-attention keys and values from it); later passes carry no source rows. Decoders of variable-length encoders see the current
/// window's rows through <see cref="InputRole.RowSpans"/> (the cached rows of longer windows stay behind them). Not thread-safe.
/// </summary>
internal sealed class SpeechEncoderDecoder : IDisposable
{
    private readonly SpeechEncoder _encoder;
    private readonly CompiledModel _decoder;
    private readonly Value _tokens;
    private readonly Value? _positions;
    private readonly Value _encoderOutput;
    private readonly Value? _outputRows;
    private readonly Value? _sourceSpans;
    private readonly float[] _logits;
    private readonly int _maxRows;
    private readonly RunParameters _run = new();
    private int[] _positionBuffer = [];
    private int[] _spanBuffer = [];

    /// <summary>Compiles the encoder and decoder components of <paramref name="model"/>.</summary>
    /// <param name="model">The loaded model.</param>
    public SpeechEncoderDecoder(TaskModel model)
    {
        _encoder = new SpeechEncoder(model, WellKnown.EncoderOutput);
        try
        {
            _decoder = model.CompileComponent(ModelComponentRoles.Decoder);
        }
        catch
        {
            _encoder.Dispose();
            throw;
        }

        var graph = _decoder.Graph;
        _tokens = graph.Input(InputRole.TokenIds);
        _positions = graph.TryInput(InputRole.Positions);
        _encoderOutput = graph.Input(InputRole.EncoderOutput);
        _outputRows = graph.TryInput(InputRole.OutputRows);
        _sourceSpans = graph.TryInput(InputRole.RowSpans);
        _logits = new float[(int)graph.Outputs[WellKnown.LogitsOutput].Shape[-1].Fixed];
        _maxRows = (int)_decoder.Options.MaxTokens;
        ContextLength = (int)_decoder.Options.ContextLength;
    }

    /// <summary>Feature channels per frame the encoder takes.</summary>
    public int FeatureChannels => _encoder.Channels;

    /// <summary>Feature frames per window the encoder takes (the most frames of a variable-length encoder).</summary>
    public int FeatureFrames => _encoder.MaxFrames;

    /// <summary>True when the encoder takes windows of any length up to <see cref="FeatureFrames"/> (no padding needed).</summary>
    public bool VariableLength => _encoder.VariableLength;

    /// <summary>The fewest frames that make at least one encoder row (shorter windows must be padded).</summary>
    public int MinFeatureFrames => _encoder.MinFrames;

    /// <summary>Encoder output rows of a full window.</summary>
    public int MaxSourceRows => _encoder.MaxRows;

    /// <summary>Encoder output rows of the current window.</summary>
    public int SourceRows => _encoder.Rows;

    /// <summary>Longest token sequence (prompt plus generated text) of one window.</summary>
    public int ContextLength { get; }

    /// <summary>Tokens of the current window already in the decoder's KV cache.</summary>
    public int Position { get; private set; }

    /// <summary>Encodes one window and starts a new decoder context on it.</summary>
    /// <param name="window">Channel-major features [<see cref="FeatureChannels"/>, frames].</param>
    /// <param name="frames">The window's frames (see <see cref="SpeechEncoder.Encode"/>).</param>
    public void Encode(ReadOnlySpan<float> window, int frames)
    {
        _encoder.Encode(window, frames);
        Position = 0;
    }

    /// <summary>
    /// Feeds tokens to the decoder and returns the logits after the last one (the array is reused by the next call). The first call
    /// after <see cref="Encode"/>, or after <see cref="Rewind"/> to 0, also hands the decoder the window's encoder output.
    /// </summary>
    /// <param name="tokens">The tokens (prompt or the last generated token).</param>
    /// <exception cref="InvalidOperationException">The tokens do not fit the decoder context.</exception>
    public float[] Decode(ReadOnlySpan<int> tokens)
    {
        if (tokens.IsEmpty)
        {
            throw new ArgumentException("No tokens.", nameof(tokens));
        }

        if (Position + tokens.Length > ContextLength)
        {
            throw new InvalidOperationException($"Speech decoder context full ({Position} + {tokens.Length} > {ContextLength}).");
        }

        while (!tokens.IsEmpty)
        {
            int rows = Math.Min(tokens.Length, _maxRows);
            Run(tokens[..rows], wantLogits: rows == tokens.Length);
            tokens = tokens[rows..];
        }

        return _logits;
    }

    /// <summary>Moves the decoder back to an earlier position of the current window (later tokens are overwritten).</summary>
    /// <param name="position">The new position, at most <see cref="Position"/>.</param>
    public void Rewind(int position)
    {
        ArgumentOutOfRangeException.ThrowIfGreaterThan(position, Position);
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        Position = position;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _decoder.Dispose();
        _encoder.Dispose();
    }

    private void Run(ReadOnlySpan<int> tokens, bool wantLogits)
    {
        int rows = tokens.Length;
        _decoder.SetInput(_tokens, tokens);
        if (_positions is not null)
        {
            if (_positionBuffer.Length < rows)
            {
                _positionBuffer = new int[Math.Max(rows, _maxRows)];
            }

            for (int i = 0; i < rows; i++)
            {
                _positionBuffer[i] = Position + i;
            }

            _decoder.SetInput<int>(_positions, _positionBuffer.AsSpan(0, rows));
        }

        // The cross-attention keys and values are projected from the source once per window, on its first pass.
        int sourceRows = Position == 0 ? SourceRows : 0;
        _decoder.SetInput<float>(_encoderOutput, _encoder.Output[..(sourceRows * _encoder.Width)]);
        if (_sourceSpans is not null)
        {
            if (_spanBuffer.Length < 2 * rows)
            {
                _spanBuffer = new int[2 * Math.Max(rows, _maxRows)];
            }

            for (int i = 0; i < rows; i++)
            {
                _spanBuffer[2 * i] = 0;
                _spanBuffer[(2 * i) + 1] = SourceRows;
            }

            _decoder.SetInput<int>(_sourceSpans, _spanBuffer.AsSpan(0, 2 * rows));
        }

        if (_outputRows is not null)
        {
            _decoder.SetInput<int>(_outputRows, [rows - 1]);
        }

        _run.Symbols[WellKnown.TokenSymbol] = rows;
        _run.Symbols[WellKnown.EncoderSymbol] = sourceRows;
        _run.Symbols["R"] = 1;
        _run.Scalars[WellKnown.PastScalar] = Position;
        _run.Scalars[WellKnown.SourcePastScalar] = 0;
        _run.SkipOutputs.Clear();
        if (!wantLogits)
        {
            _run.SkipOutputs.Add(WellKnown.LogitsOutput);
        }

        _decoder.Run(_run);
        Position += rows;
        if (wantLogits)
        {
            _decoder.GetOutput<float>(WellKnown.LogitsOutput, _logits);
        }
    }
}
