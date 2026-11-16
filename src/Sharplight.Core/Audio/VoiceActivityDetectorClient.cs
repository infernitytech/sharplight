using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.Audio;

/// <summary>The built-in <see cref="IVoiceActivityDetector"/> over the <see cref="VoiceActivityGraphs"/> contract.</summary>
internal sealed class VoiceActivityDetectorClient : IVoiceActivityDetector
{
    private readonly TaskModel _model;
    private readonly CompiledModel _compiled;
    private readonly Value _audio;
    private readonly Value _state;
    private readonly int _context;
    private readonly int _stateLength;
    private bool _disposed;

    public VoiceActivityDetectorClient(TaskModel model)
    {
        _model = model;
        var component = model.Description.Component(ModelComponentRoles.Encoder);
        SampleRate = component.Config[VoiceActivityGraphs.SampleRateKey].GetInt32();
        FrameSamples = component.Config[VoiceActivityGraphs.FrameSamplesKey].GetInt32();
        _context = component.Config[VoiceActivityGraphs.ContextSamplesKey].GetInt32();
        _compiled = model.CompileComponent(ModelComponentRoles.Encoder, VoiceActivityGraphs.StepGraph);
        _audio = _compiled.Graph.Inputs.First(v => v.Input!.Name == VoiceActivityGraphs.AudioInput);
        _state = _compiled.Graph.Inputs.First(v => v.Input!.Name == VoiceActivityGraphs.StateInput);
        _stateLength = (int)_state.Shape.Elements();
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    public TaskClientMetadata Metadata { get; }

    public int SampleRate { get; }

    public int FrameSamples { get; }

    public object? GetService(Type serviceType, object? serviceKey = null)
        => serviceKey is null && serviceType.IsInstanceOfType(_model) ? _model : null;

    public Task<float[]> ProbabilitiesAsync(AudioBuffer audio, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        ObjectDisposedException.ThrowIf(_disposed, this);
        var samples = audio.ToModelInput(SampleRate).Samples;
        return Task.Run(() => Probabilities(samples, cancellationToken), cancellationToken);
    }

    public async Task<IReadOnlyList<SpeechSegment>> DetectAsync(AudioBuffer audio, VoiceActivityOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        var probabilities = await ProbabilitiesAsync(audio, cancellationToken).ConfigureAwait(false);
        long total = (long)Math.Round(audio.FrameCount * (double)SampleRate / audio.SampleRate);
        var segments = SpeechSegmentation.FromProbabilities(probabilities, FrameSamples, SampleRate, total, options);
        if (audio.SampleRate == SampleRate)
        {
            return segments;
        }

        double scale = (double)audio.SampleRate / SampleRate;
        return [.. segments.Select(s => new SpeechSegment((long)Math.Round(s.StartSample * scale), Math.Min(audio.FrameCount, (long)Math.Round(s.EndSample * scale)), audio.SampleRate))];
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _compiled.Dispose();
    }

    private float[] Probabilities(float[] samples, CancellationToken cancellationToken)
    {
        int frames = (samples.Length + FrameSamples - 1) / FrameSamples;
        var probabilities = new float[frames];
        var state = new float[_stateLength];
        var window = new float[_context + FrameSamples];
        var output = new float[1];
        var parameters = new RunParameters();
        for (int i = 0; i < frames; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int start = i * FrameSamples;
            if (i == 0)
            {
                Array.Clear(window, 0, _context);
            }
            else
            {
                Array.Copy(window, FrameSamples, window, 0, _context);
            }

            int count = Math.Min(FrameSamples, samples.Length - start);
            Array.Copy(samples, start, window, _context, count);
            Array.Clear(window, _context + count, FrameSamples - count);
            _compiled.SetInput<float>(_audio, window);
            _compiled.SetInput<float>(_state, state);
            _compiled.Run(parameters);
            _compiled.GetOutput<float>(VoiceActivityGraphs.ProbabilityOutput, output);
            _compiled.GetOutput<float>(VoiceActivityGraphs.StateOutput, state);
            probabilities[i] = output[0];
        }

        return probabilities;
    }
}
