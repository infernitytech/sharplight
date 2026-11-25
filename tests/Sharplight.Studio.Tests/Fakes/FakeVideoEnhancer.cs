using System.Runtime.CompilerServices;
using Sharplight.Tasks;
using Sharplight.VideoEnhancement;
using Sharplight.Vision;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>A video enhancer that streams the stages of each batch and resizes the frames to the requested size; records what it was asked.</summary>
internal sealed class FakeVideoEnhancer : IVideoEnhancer
{
    /// <summary>The defaults the model reports.</summary>
    public VideoEnhancementDefaults Defaults { get; set; } = new(3f, 16, 4, VideoColorCorrection.Wavelet, 5);

    /// <summary>The requests and options received.</summary>
    public List<(VideoEnhancementRequest Request, VideoEnhancementOptions Options)> Requests { get; } = [];

    /// <summary>When set, the stream waits on it after the first stage (until cancelled).</summary>
    public TaskCompletionSource? Gate { get; set; }

    public TaskClientMetadata Metadata { get; } = new(Sharplight.Models.ModelKinds.VideoEnhancement, "fake-enhancer", "fake");

    public Task<VideoEnhancementResult> EnhanceAsync(VideoEnhancementRequest request, VideoEnhancementOptions? options = null, CancellationToken cancellationToken = default)
        => throw new NotSupportedException("The page streams.");

    public async IAsyncEnumerable<VideoEnhancementUpdate> EnhanceStreamingAsync(VideoEnhancementRequest request, VideoEnhancementOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        options ??= VideoEnhancementOptions.Default;
        Requests.Add((request, options));
        var input = request.Frames[0];
        int width = options.Width ?? (int)(input.Width * (options.Scale ?? Defaults.Scale));
        int height = options.Height ?? (int)(input.Height * (options.Scale ?? Defaults.Scale));
        int perBatch = options.FramesPerBatch is > 0 and var frames ? frames : request.Frames.Count;
        int batches = (request.Frames.Count + perBatch - 1) / perBatch;
        var output = new List<ImageInput>();
        for (int batch = 0; batch < batches; batch++)
        {
            yield return new VideoEnhancementUpdate(VideoEnhancementPhase.Encoding, batch, batches);
            await Task.Yield();
            cancellationToken.ThrowIfCancellationRequested();
            yield return new VideoEnhancementUpdate(VideoEnhancementPhase.Restoring, batch, batches);
            if (batch == 0 && Gate is { } gate)
            {
                await gate.Task.WaitAsync(cancellationToken);
            }

            int count = Math.Min(perBatch, request.Frames.Count - batch * perBatch);
            var decoded = Enumerable.Range(0, count).Select(i => FakeImageGenerator.Gradient(width, height, (batch * perBatch + i) * 10)).ToList();
            output.AddRange(decoded);
            yield return new VideoEnhancementUpdate(VideoEnhancementPhase.Decoding, batch, batches, Frames: decoded);
        }

        var result = new VideoEnhancementResult(output, request.FramesPerSecond ?? 24, options.Seed ?? 777);
        yield return new VideoEnhancementUpdate(VideoEnhancementPhase.Completed, batches - 1, batches, Result: result);
    }

    public object? GetService(Type serviceType, object? serviceKey = null) => null;

    public void Dispose()
    {
    }
}
