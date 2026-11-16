using System.Runtime.CompilerServices;
using Sharplight.Tasks;
using Sharplight.VideoGeneration;
using Sharplight.Vision;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>A video generator that streams the phases with previews and decoded frames in chunks; records what it was asked.</summary>
internal sealed class FakeVideoGenerator : IVideoGenerator
{
    /// <summary>The defaults the model reports.</summary>
    public VideoGenerationDefaults Defaults { get; set; } = new(64, 48, 9, 12, 4, 5f, "unipc", 16, 4, true, "blurry, static");

    /// <summary>The requests and options received.</summary>
    public List<(VideoGenerationRequest Request, VideoGenerationOptions Options)> Requests { get; } = [];

    /// <summary>When set, the stream waits on it after the first step (until cancelled).</summary>
    public TaskCompletionSource? Gate { get; set; }

    public TaskClientMetadata Metadata { get; } = new(Sharplight.Models.ModelKinds.VideoGeneration, "fake-wan", "wan");

    public Task<VideoGenerationResult> GenerateAsync(VideoGenerationRequest request, VideoGenerationOptions? options = null, CancellationToken cancellationToken = default)
        => throw new NotSupportedException("The page streams.");

    public async IAsyncEnumerable<VideoGenerationUpdate> GenerateStreamingAsync(VideoGenerationRequest request, VideoGenerationOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        options ??= VideoGenerationOptions.Default;
        Requests.Add((request, options));
        yield return new VideoGenerationUpdate(VideoGenerationPhase.Conditioning, 0, 0);
        int steps = options.Steps ?? 4;
        for (int step = 1; step <= steps; step++)
        {
            await Task.Yield();
            cancellationToken.ThrowIfCancellationRequested();
            var preview = options.PreviewInterval > 0 && step % options.PreviewInterval == 0 ? new[] { Frame(8, 8, step) } : null;
            yield return new VideoGenerationUpdate(VideoGenerationPhase.Denoising, step, steps, Preview: preview);
            if (step == 1 && Gate is { } gate)
            {
                await gate.Task.WaitAsync(cancellationToken);
            }
        }

        int width = options.Width ?? Defaults.Width;
        int height = options.Height ?? Defaults.Height;
        int count = options.Frames ?? Defaults.Frames;
        var frames = Enumerable.Range(0, count).Select(i => Frame(width, height, i * 20)).ToList();
        for (int start = 0; start < count; start += 4)
        {
            yield return new VideoGenerationUpdate(VideoGenerationPhase.Decoding, Math.Min(count, start + 4), count, Frames: frames.Skip(start).Take(4).ToList());
        }

        long seed = options.Seed ?? 424242;
        var result = new VideoGenerationResult(frames, options.FramesPerSecond ?? Defaults.FramesPerSecond, seed);
        yield return new VideoGenerationUpdate(VideoGenerationPhase.Completed, steps, steps, result);
    }

    public object? GetService(Type serviceType, object? serviceKey = null) => serviceType == typeof(VideoGenerationDefaults) ? Defaults : null;

    public void Dispose()
    {
    }

    private static ImageInput Frame(int width, int height, int tint) => FakeImageGenerator.Gradient(width, height, tint);
}
