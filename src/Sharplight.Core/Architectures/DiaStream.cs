using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.Architectures;

/// <summary>
/// One decoder stream of Dia: a compiled decoder with its own caches, the conditioning rows of its text, and the position reached. Classifier-free guidance runs two
/// streams over the same decoder rows, one on the text and one on a text of zeros.
/// </summary>
internal sealed class DiaStream : IDisposable
{
    private readonly CompiledModel _decoder;
    private readonly Value[] _codes;
    private readonly Value _positions;
    private readonly Value _condition;
    private readonly Value _spans;
    private readonly int _channels;
    private readonly int _vocabulary;
    private readonly int _width;
    private readonly float[] _logits;
    private float[] _source = [];
    private int _sourceRows;

    /// <summary>Compiles the decoder step graph of <paramref name="model"/>.</summary>
    /// <param name="model">A loaded Dia model.</param>
    /// <param name="channels">Codebooks.</param>
    /// <param name="vocabulary">Entries per codebook head.</param>
    /// <param name="width">The width of the conditioning rows.</param>
    public DiaStream(TaskModel model, int channels, int vocabulary, int width)
    {
        _decoder = model.CompileComponent(DiaGraphs.DecoderRole, DiaGraphs.StepGraph);
        _channels = channels;
        _vocabulary = vocabulary;
        _width = width;
        var inputs = _decoder.Graph.Inputs;
        _codes = [.. Enumerable.Range(0, channels).Select(c => inputs.First(v => v.Input!.Name == DiaGraphs.CodeInput(c)))];
        _positions = inputs.First(v => v.Input!.Name == DiaGraphs.PositionsInput);
        _condition = inputs.First(v => v.Input!.Name == DiaGraphs.ConditionOutput);
        _spans = inputs.First(v => v.Input!.Name == DiaGraphs.SpansInput);
        _logits = new float[DiaAdapter.MaxRows * channels * vocabulary];
    }

    /// <summary>The rows of the decoder's sequence processed so far.</summary>
    public int Position { get; private set; }

    /// <summary>Starts a new sequence on the conditioning rows <paramref name="source"/> (the next <see cref="Process"/> projects them into the cross-attention states).</summary>
    /// <param name="source">The encoder output [rows, width].</param>
    public void Begin(float[] source)
    {
        _source = source;
        _sourceRows = source.Length / _width;
        Position = 0;
    }

    /// <summary>Feeds rows to the decoder (any number; longer runs go in passes) and returns the logits of the last one.</summary>
    /// <param name="rows">The rows, each with a token per codebook.</param>
    /// <param name="first">The first row to feed.</param>
    /// <param name="count">The number of rows.</param>
    /// <returns>The last row's logits [channels · vocabulary] (reused by the next call).</returns>
    public ReadOnlySpan<float> Process(IReadOnlyList<int[]> rows, int first, int count)
    {
        int width = _channels * _vocabulary;
        int done = 0;
        while (done < count)
        {
            int n = Math.Min(DiaAdapter.MaxRows, count - done);
            for (int c = 0; c < _channels; c++)
            {
                var ids = new int[n];
                for (int i = 0; i < n; i++)
                {
                    ids[i] = rows[first + done + i][c] + (c * _vocabulary);
                }

                _decoder.SetInput<int>(_codes[c], ids);
            }

            var positions = new int[n];
            var spans = new int[2 * n];
            for (int i = 0; i < n; i++)
            {
                positions[i] = Position + i;
                spans[(2 * i) + 1] = _sourceRows;
            }

            _decoder.SetInput<int>(_positions, positions);
            _decoder.SetInput<int>(_spans, spans);
            int sourceRows = Position == 0 ? _sourceRows : 0;
            _decoder.SetInput<float>(_condition, _source.AsSpan(0, sourceRows * _width));
            var parameters = new RunParameters();
            parameters.Symbols[WellKnown.TokenSymbol] = n;
            parameters.Symbols[WellKnown.EncoderSymbol] = sourceRows;
            parameters.Scalars[WellKnown.PastScalar] = Position;
            parameters.Scalars[WellKnown.SourcePastScalar] = 0;
            _decoder.Run(parameters);
            _decoder.GetOutput<float>(DiaGraphs.LogitsOutput, _logits.AsSpan(0, n * width));
            Position += n;
            done += n;
            if (done == count)
            {
                return _logits.AsSpan((n - 1) * width, width);
            }
        }

        throw new ArgumentException("No rows.", nameof(count));
    }

    /// <inheritdoc/>
    public void Dispose() => _decoder.Dispose();
}
