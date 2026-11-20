using Sharplight.Audio;
using Sharplight.Tasks;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>A diffusion audio generator (ACE-Step by default; the traits are settable for Stable Audio) that returns a tone and records the requests.</summary>
internal sealed class FakeAudioGenerator : IAudioGenerator
{
    public TaskClientMetadata Metadata { get; } = new(Sharplight.Models.ModelKinds.AudioGeneration, "fake-acestep", "acestep");

    public int SampleRate => 8000;

    public int Channels => 1;

    public double LatentFrameRate => 25;

    public int LatentChannels => 64;

    public TimeSpan MaxDuration { get; set; } = TimeSpan.FromSeconds(60);

    public int DefaultSteps { get; set; } = 8;

    public bool SupportsLyrics { get; set; } = true;

    public bool SupportsNegativePrompt { get; set; }

    public float? DefaultGuidanceScale { get; set; }

    public bool SupportsShift { get; set; } = true;

    /// <summary>The requests received.</summary>
    public List<AudioGenerationRequest> Requests { get; } = [];

    public Task<AudioLatents> GenerateLatentsAsync(AudioGenerationRequest request, IProgress<(int Step, int Steps)>? progress = null, CancellationToken cancellationToken = default)
        => throw new NotSupportedException();

    public Task<AudioBuffer> GenerateAsync(AudioGenerationRequest request, IProgress<(int Step, int Steps)>? progress = null, CancellationToken cancellationToken = default)
    {
        Requests.Add(request);
        progress?.Report((1, 1));
        return Task.FromResult(new AudioBuffer([.. Enumerable.Range(0, (int)(SampleRate * request.Duration.TotalSeconds)).Select(i => 0.2f * MathF.Sin(2 * MathF.PI * 330 * i / SampleRate))], SampleRate));
    }

    public object? GetService(Type serviceType, object? serviceKey = null) => null;

    public void Dispose()
    {
    }
}
