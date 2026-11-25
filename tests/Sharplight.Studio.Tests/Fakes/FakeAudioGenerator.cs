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

    public AudioGenerationDefaults Defaults => new(DefaultSteps, DefaultGuidanceScale, SupportsLyrics, SupportsNegativePrompt, SupportsShift);

    public int DefaultSteps { get; set; } = 8;

    public bool SupportsLyrics { get; set; } = true;

    public bool SupportsNegativePrompt { get; set; }

    public float? DefaultGuidanceScale { get; set; }

    public bool SupportsShift { get; set; } = true;

    /// <summary>The requests received.</summary>
    public List<AudioGenerationRequest> Requests { get; } = [];

    /// <summary>The options of the requests received.</summary>
    public List<AudioGenerationOptions?> Options { get; } = [];

    public Task<AudioLatents> GenerateLatentsAsync(AudioGenerationRequest request, AudioGenerationOptions? options = null, CancellationToken cancellationToken = default)
        => throw new NotSupportedException();

    public Task<AudioGenerationResult> GenerateAsync(AudioGenerationRequest request, AudioGenerationOptions? options = null, CancellationToken cancellationToken = default)
        => Task.FromResult(Generate(request, options));

    public async IAsyncEnumerable<AudioGenerationUpdate> GenerateStreamingAsync(AudioGenerationRequest request, AudioGenerationOptions? options = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await Task.Yield();
        yield return new AudioGenerationUpdate(1, 1, Generate(request, options));
    }

    private AudioGenerationResult Generate(AudioGenerationRequest request, AudioGenerationOptions? options)
    {
        Requests.Add(request);
        Options.Add(options);
        var audio = new AudioBuffer([.. Enumerable.Range(0, (int)(SampleRate * request.Duration.TotalSeconds)).Select(i => 0.2f * MathF.Sin(2 * MathF.PI * 330 * i / SampleRate))], SampleRate);
        return new AudioGenerationResult(audio, options?.Seed ?? 1);
    }

    public object? GetService(Type serviceType, object? serviceKey = null) => null;

    public void Dispose()
    {
    }
}
