using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.Audio;

/// <summary>The built-in <see cref="IAudioCodec"/> over the <see cref="AudioCodecGraphs"/> contract.</summary>
internal sealed class AudioCodecClient : IAudioCodec
{
    private readonly TaskModel _model;
    private readonly CompiledModel _encoder;
    private readonly CompiledModel _decoder;
    private readonly Value _audio;
    private readonly Value[] _decoderCodes;
    private readonly Value _mask;
    private readonly int _maxFrames;
    private readonly Value? _encoderPositions;
    private readonly Value? _decoderPositions;
    private readonly Value? _encoderSpans;
    private readonly Value? _decoderSpans;
    private readonly (int Numerator, int Denominator) _encoderRows;
    private readonly (int Numerator, int Denominator) _decoderRows;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private bool _disposed;

    public AudioCodecClient(TaskModel model)
    {
        _model = model;
        var encoder = model.Description.Component(ModelComponentRoles.Encoder);
        var decoder = model.Description.Component(ModelComponentRoles.Decoder);
        SampleRate = encoder.Config[AudioCodecGraphs.SampleRateKey].GetInt32();
        SamplesPerFrame = encoder.Config[AudioCodecGraphs.SamplesPerFrameKey].GetInt32();
        CodebookCount = encoder.Config[AudioCodecGraphs.CodebooksKey].GetInt32();
        DefaultCodebooks = encoder.Config[AudioCodecGraphs.DefaultCodebooksKey].GetInt32();
        CodebookSize = encoder.Config[AudioCodecGraphs.CodebookSizeKey].GetInt32();
        _maxFrames = (int)decoder.SymbolBounds.GetValueOrDefault(WellKnown.TokenSymbol, 1024);
        _encoder = model.CompileComponent(ModelComponentRoles.Encoder, AudioCodecGraphs.EncodeGraph);
        _decoder = model.CompileComponent(ModelComponentRoles.Decoder, AudioCodecGraphs.DecodeGraph);
        _audio = _encoder.Graph.Inputs.First(v => v.Input!.Name == AudioCodecGraphs.AudioInput);
        _mask = _decoder.Graph.Inputs.First(v => v.Input!.Name == AudioCodecGraphs.MaskInput);
        _decoderCodes = [.. Enumerable.Range(0, CodebookCount).Select(q => _decoder.Graph.Inputs.First(v => v.Input!.Name == AudioCodecGraphs.CodesName(q)))];
        _encoderPositions = _encoder.Graph.Inputs.FirstOrDefault(v => v.Input!.Name == AudioCodecGraphs.PositionsInput);
        _decoderPositions = _decoder.Graph.Inputs.FirstOrDefault(v => v.Input!.Name == AudioCodecGraphs.PositionsInput);
        _encoderSpans = _encoder.Graph.Inputs.FirstOrDefault(v => v.Input!.Name == AudioCodecGraphs.SpansInput);
        _decoderSpans = _decoder.Graph.Inputs.FirstOrDefault(v => v.Input!.Name == AudioCodecGraphs.SpansInput);
        _encoderRows = Rows(encoder);
        _decoderRows = Rows(decoder);
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    public TaskClientMetadata Metadata { get; }

    private static (int, int) Rows(ModelComponent component)
        => (component.Config.TryGetValue(AudioCodecGraphs.PositionsNumeratorKey, out var n) ? n.GetInt32() : 1,
            component.Config.TryGetValue(AudioCodecGraphs.PositionsDenominatorKey, out var d) ? d.GetInt32() : 1);

    // The row positions and the single-sequence attention spans of a transformer inside the graph, when it has them.
    private static void SetPositions(CompiledModel compiled, Value? positions, Value? spans, (int Numerator, int Denominator) rows, int length)
    {
        int count = (int)((long)length * rows.Numerator / rows.Denominator);
        if (positions is not null)
        {
            var values = new int[count];
            for (int i = 0; i < count; i++)
            {
                values[i] = i;
            }

            compiled.SetInput<int>(positions, values);
        }

        if (spans is not null)
        {
            var values = new int[2 * count];
            for (int i = 0; i < count; i++)
            {
                values[(2 * i) + 1] = count;
            }

            compiled.SetInput<int>(spans, values);
        }
    }

    public int SampleRate { get; }

    public int SamplesPerFrame { get; }

    public double FrameRate => (double)SampleRate / SamplesPerFrame;

    public int CodebookCount { get; }

    public int DefaultCodebooks { get; }

    public int CodebookSize { get; }

    public object? GetService(Type serviceType, object? serviceKey = null)
        => serviceKey is null && serviceType.IsInstanceOfType(_model) ? _model : null;

    public Task<CodecCodes> EncodeAsync(AudioBuffer audio, AudioCodecOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        ObjectDisposedException.ThrowIf(_disposed, this);
        int codebooks = Math.Clamp(options?.Codebooks ?? DefaultCodebooks, 1, CodebookCount);
        return BackgroundCompute.RunAsync(_gate, () => Encode(audio.ToModelInput(SampleRate).Samples, codebooks, cancellationToken), cancellationToken);
    }

    public Task<AudioBuffer> DecodeAsync(CodecCodes codes, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(codes);
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (codes.Codebooks < 1 || codes.Codebooks > CodebookCount)
        {
            throw new ArgumentException($"The codec has {CodebookCount} codebooks; the codes have {codes.Codebooks}.", nameof(codes));
        }

        return BackgroundCompute.RunAsync(_gate, () => Decode(codes, cancellationToken), cancellationToken);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _encoder.Dispose();
        _decoder.Dispose();
        _gate.Dispose();
    }

    private CodecCodes Encode(float[] samples, int codebooks, CancellationToken cancellationToken)
    {
        int frames = (samples.Length + SamplesPerFrame - 1) / SamplesPerFrame;
        var result = new int[codebooks * frames];
        for (int start = 0; start < frames; start += _maxFrames)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int count = Math.Min(_maxFrames, frames - start);
            var segment = new float[count * SamplesPerFrame];
            int from = start * SamplesPerFrame;
            samples.AsSpan(from, Math.Min(segment.Length, samples.Length - from)).CopyTo(segment);
            _encoder.SetInput<float>(_audio, segment);
            SetPositions(_encoder, _encoderPositions, _encoderSpans, _encoderRows, segment.Length);
            var parameters = new RunParameters();
            parameters.Symbols[WellKnown.TokenSymbol] = segment.Length;
            _encoder.Run(parameters);
            var part = new int[count];
            for (int q = 0; q < codebooks; q++)
            {
                _encoder.GetOutput<int>(AudioCodecGraphs.CodesName(q), part);
                part.CopyTo(result.AsSpan((q * frames) + start, count));
            }
        }

        return new CodecCodes(codebooks, frames, result);
    }

    private AudioBuffer Decode(CodecCodes codes, CancellationToken cancellationToken)
    {
        var mask = new float[CodebookCount];
        mask.AsSpan(0, codes.Codebooks).Fill(1f);
        var audio = new float[codes.Frames * SamplesPerFrame];
        for (int start = 0; start < codes.Frames; start += _maxFrames)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int count = Math.Min(_maxFrames, codes.Frames - start);
            _decoder.SetInput<float>(_mask, mask);
            SetPositions(_decoder, _decoderPositions, _decoderSpans, _decoderRows, count);
            var zero = new int[count];
            for (int q = 0; q < CodebookCount; q++)
            {
                _decoder.SetInput<int>(_decoderCodes[q], q < codes.Codebooks ? codes.Codebook(q).Slice(start, count).ToArray() : zero);
            }

            var parameters = new RunParameters();
            parameters.Symbols[WellKnown.TokenSymbol] = count;
            _decoder.Run(parameters);
            var part = new float[count * SamplesPerFrame];
            _decoder.GetOutput<float>(AudioCodecGraphs.AudioOutput, part);
            part.CopyTo(audio.AsSpan(start * SamplesPerFrame));
        }

        return new AudioBuffer(audio, SampleRate);
    }
}
