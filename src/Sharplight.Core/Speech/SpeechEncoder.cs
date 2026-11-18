using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.Tensors;

namespace Sharplight.Speech;

/// <summary>
/// Runs the compiled encoder graph of a speech model over one window of audio features (<see cref="InputRole.AudioFeatures"/>) and keeps its
/// output rows. Fixed-window encoders (Whisper) take channel-major features [channels, frames]; variable-length encoders take time-major
/// features [<see cref="WellKnown.AudioFrameSymbol"/>, channels] of up to the component's bound of frames. An optional
/// <see cref="InputRole.Positions"/> input [P, 1] gets the row positions 0 … rows − 1 when P is the row count (rotary attention) or the
/// relative distances rows − 1 … −(rows − 1) when P is 2·rows − 1 (Transformer-XL relative attention). Not thread-safe.
/// </summary>
internal sealed class SpeechEncoder : IDisposable
{
    private readonly CompiledModel _encoder;
    private readonly Value _features;
    private readonly Value? _positions;
    private readonly Value? _spans;
    private readonly Value? _prompt;
    private readonly int _chunkRows;
    private readonly int _leftChunks;
    private float[]? _promptValues;
    private readonly Dictionary<string, int> _promptDictionary = new(StringComparer.OrdinalIgnoreCase);
    private readonly int _promptDefault;
    private readonly string _output;
    private readonly Dim _rows;
    private readonly Dim _positionRows;
    private readonly float[] _encoded;
    private readonly RunParameters _run = new();
    private int[] _positionBuffer = [];
    private float[] _timeMajor = [];

    /// <summary>Compiles the encoder component of <paramref name="model"/>.</summary>
    /// <param name="model">The loaded model.</param>
    /// <param name="output">The output to keep (<see cref="WellKnown.EncoderOutput"/>, or <see cref="WellKnown.LogitsOutput"/> of CTC models).</param>
    /// <exception cref="NotSupportedException">The encoder input is not two-dimensional.</exception>
    public SpeechEncoder(TaskModel model, string output)
    {
        _encoder = model.CompileComponent(ModelComponentRoles.Encoder);
        try
        {
            _output = output;
            _features = _encoder.Graph.Input(InputRole.AudioFeatures);
            if (_features.Rank != 2)
            {
                throw new NotSupportedException($"The speech encoder input must be [channels, frames] or [frames, channels], got {_features.Shape}.");
            }

            VariableLength = _features.Shape[0].IsSymbolic;
            Channels = (int)_features.Shape[VariableLength ? 1 : 0].Fixed;
            MaxFrames = VariableLength ? (int)_encoder.Options.Bound(WellKnown.AudioFrameSymbol) : (int)_features.Shape[1].Fixed;
            _positions = _encoder.Graph.TryInput(InputRole.Positions);
            _positionRows = _positions?.Shape[0] ?? Dim.Of(0);
            var component = model.Description.Component(ModelComponentRoles.Encoder);
            _spans = _encoder.Graph.TryInput(InputRole.RowSpans);
            _chunkRows = Math.Max(component.ConfigValue("chunkRows", 1), 1);
            _leftChunks = component.ConfigValue("leftContext", 0) / _chunkRows;
            _prompt = _encoder.Graph.Inputs.FirstOrDefault(v => v.Input!.Name == "prompt");
            PromptCount = _prompt is null ? 0 : (int)_prompt.Shape[1].Fixed;
            _promptDefault = component.ConfigValue("promptDefault", 0);
            PromptIndex = _promptDefault;
            foreach (string pair in component.ConfigValue("promptDictionary", "").Split(',', StringSplitOptions.RemoveEmptyEntries))
            {
                int at = pair.IndexOf('=', StringComparison.Ordinal);
                if (at > 0 && int.TryParse(pair[(at + 1)..], System.Globalization.CultureInfo.InvariantCulture, out int index))
                {
                    _promptDictionary[pair[..at]] = index;
                }
            }
            var shape = _encoder.Graph.Outputs[output].Shape;
            _rows = shape[0];
            Width = (int)shape[1].Fixed;
            MaxRows = (int)RowsOf(MaxFrames);
            Rows = MaxRows;
            MinFrames = VariableLength ? SmallestWindow() : MaxFrames;
            _encoded = new float[MaxRows * Width];
        }
        catch
        {
            _encoder.Dispose();
            throw;
        }
    }

    /// <summary>The size of the language prompt one-hot of the encoder (0 when it has no prompt).</summary>
    public int PromptCount { get; }

    /// <summary>The language prompt index of the next windows (see <see cref="PromptCount"/>).</summary>
    public int PromptIndex { get; set; }

    /// <summary>Selects the language prompt for a request: the language's entry of the model's prompt dictionary (a locale such as "de-DE" or a bare code), the default (automatic detection) without one.</summary>
    /// <param name="language">The requested language, or <see langword="null"/>.</param>
    /// <exception cref="NotSupportedException">The model has no prompt for the language.</exception>
    public void SelectLanguage(string? language)
    {
        if (_prompt is null)
        {
            return;
        }

        if (string.IsNullOrWhiteSpace(language))
        {
            PromptIndex = _promptDefault;
            return;
        }

        PromptIndex = _promptDictionary.TryGetValue(language, out int index) || _promptDictionary.TryGetValue(SpeechLanguages.Code(language), out index)
            ? index
            : throw new NotSupportedException($"The model has no language prompt for '{language}'.");
    }

    /// <summary>Feature channels per frame.</summary>
    public int Channels { get; }

    /// <summary>Frames of a full window (the most frames of a variable-length encoder).</summary>
    public int MaxFrames { get; }

    /// <summary>The fewest frames that make at least one row (shorter windows must be padded).</summary>
    public int MinFrames { get; }

    /// <summary>True when the encoder takes windows of any length from <see cref="MinFrames"/> to <see cref="MaxFrames"/>.</summary>
    public bool VariableLength { get; }

    /// <summary>Output rows of a full window.</summary>
    public int MaxRows { get; }

    /// <summary>Output rows of the last window.</summary>
    public int Rows { get; private set; }

    /// <summary>Values per output row.</summary>
    public int Width { get; }

    /// <summary>The output rows of the last window, row-major [<see cref="Rows"/>, <see cref="Width"/>].</summary>
    public ReadOnlySpan<float> Output => _encoded.AsSpan(0, Rows * Width);

    /// <summary>The output rows of the last window as an array segment (valid until the next <see cref="Encode"/>).</summary>
    public ArraySegment<float> OutputSegment => new(_encoded, 0, Rows * Width);

    /// <summary>Encodes one window.</summary>
    /// <param name="window">Channel-major features [<see cref="Channels"/>, frames].</param>
    /// <param name="frames">The window's frames: <see cref="MaxFrames"/> for fixed-window encoders, <see cref="MinFrames"/> to <see cref="MaxFrames"/> otherwise.</param>
    /// <exception cref="ArgumentOutOfRangeException">The frame count does not fit the encoder.</exception>
    public void Encode(ReadOnlySpan<float> window, int frames)
    {
        if (frames > MaxFrames || frames < MinFrames || (!VariableLength && frames != MaxFrames))
        {
            throw new ArgumentOutOfRangeException(nameof(frames), frames, $"The speech encoder takes {MinFrames} to {MaxFrames} frames.");
        }

        _run.Symbols.Clear();
        _run.SkipOutputs.Clear();
        foreach (string name in _encoder.Graph.Outputs.Keys.Where(name => name != _output))
        {
            _run.SkipOutputs.Add(name);
        }

        if (VariableLength)
        {
            Rows = (int)RowsOf(frames);
            _run.Symbols[WellKnown.AudioFrameSymbol] = frames;
            _encoder.SetInput(_features, TimeMajor(window, frames));
            if (_positions is not null)
            {
                _encoder.SetInput<int>(_positions, Positions(frames));
            }

            if (_spans is not null)
            {
                _encoder.SetInput<int>(_spans, ChunkSpans(Rows));
            }

            if (_prompt is not null)
            {
                _promptValues ??= new float[PromptCount];
                Array.Clear(_promptValues);
                _promptValues[PromptIndex] = 1f;
                _encoder.SetInput<float>(_prompt, _promptValues);
            }
        }
        else
        {
            _encoder.SetInput(_features, window);
        }

        _encoder.Run(_run);
        _encoder.GetOutput<float>(_output, _encoded.AsSpan(0, Rows * Width));
    }

    /// <inheritdoc/>
    public void Dispose() => _encoder.Dispose();

    private long RowsOf(int frames)
        => _rows.IsSymbolic ? _rows.Resolve(new Dictionary<string, long> { [WellKnown.AudioFrameSymbol] = frames }) : _rows.Fixed;

    // Binary search for the shortest window with one row (the row count grows with the frame count).
    private int SmallestWindow()
    {
        int low = 1;
        int high = MaxFrames;
        while (low < high)
        {
            int middle = low + ((high - low) / 2);
            if (RowsOf(middle) >= 1)
            {
                high = middle;
            }
            else
            {
                low = middle + 1;
            }
        }

        return low;
    }

    // Chunk-limited attention: a row sees its own chunk (the rows with the same index / chunk size, up to the chunk's last row) and the chunks of the left context.
    private int[] ChunkSpans(int rows)
    {
        var spans = new int[2 * rows];
        for (int i = 0; i < rows; i++)
        {
            int chunk = i / _chunkRows;
            int first = Math.Max(0, chunk - _leftChunks) * _chunkRows;
            int last = Math.Min(rows, (chunk + 1) * _chunkRows);
            spans[2 * i] = first;
            spans[(2 * i) + 1] = last - first;
        }

        return spans;
    }

    // Row positions, or relative distances when the input has 2·rows − 1 entries.
    private ReadOnlySpan<int> Positions(int frames)
    {
        int count = (int)(_positionRows.IsSymbolic
            ? _positionRows.Resolve(new Dictionary<string, long> { [WellKnown.AudioFrameSymbol] = frames })
            : _positionRows.Fixed);
        if (_positionBuffer.Length < count)
        {
            _positionBuffer = new int[count];
        }

        bool relative = count == (2 * Rows) - 1;
        for (int i = 0; i < count; i++)
        {
            _positionBuffer[i] = relative ? Rows - 1 - i : i;
        }

        return _positionBuffer.AsSpan(0, count);
    }

    // The channel-major window [C, frames] as time-major [frames, C] (a copy only when there is more than one channel).
    private ReadOnlySpan<float> TimeMajor(ReadOnlySpan<float> window, int frames)
    {
        if (Channels == 1)
        {
            return window[..frames];
        }

        if (_timeMajor.Length < Channels * MaxFrames)
        {
            _timeMajor = new float[Channels * MaxFrames];
        }

        for (int c = 0; c < Channels; c++)
        {
            var row = window.Slice(c * frames, frames);
            for (int t = 0; t < frames; t++)
            {
                _timeMajor[(t * Channels) + c] = row[t];
            }
        }

        return _timeMajor.AsSpan(0, Channels * frames);
    }
}
