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

    /// <summary>When set, awaited after each batch's finished frames were handed out (the argument is the batch).</summary>
    public Func<int, Task>? AfterBatch { get; set; }

    public TaskClientMetadata Metadata { get; } = new(Sharplight.Models.ModelKinds.VideoEnhancement, "fake-enhancer", "fake");

    /// <summary>The frames read from the requests' frames or sources.</summary>
    public int FramesRead { get; private set; }

    public async IAsyncEnumerable<VideoEnhancementUpdate> EnhanceStreamingAsync(VideoEnhancementRequest request, VideoEnhancementOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        options ??= VideoEnhancementOptions.Default;
        Requests.Add((request, options));
        await using var frames = request.ReadFramesAsync(cancellationToken).GetAsyncEnumerator(cancellationToken);
        long? known = request.KnownFrameCount;
        int perBatch = options.FramesPerBatch is > 0 and var size ? size : (int)Math.Min(int.MaxValue, known ?? int.MaxValue);
        int batches = known is { } count ? (int)((count + perBatch - 1) / perBatch) : 0;
        long seed = options.Seed ?? 777;
        double rate = request.FramesPerSecond ?? 24;
        long finished = 0;
        int batch = 0;
        (int Width, int Height) output = default;
        bool more = true;
        while (more)
        {
            var input = new List<ImageInput>();
            while (input.Count < perBatch && (more = await frames.MoveNextAsync()))
            {
                input.Add(frames.Current);
                FramesRead++;
            }

            if (input.Count == 0)
            {
                break;
            }

            int width = options.Width ?? (int)(input[0].Width * (options.Scale ?? Defaults.Scale));
            int height = options.Height ?? (int)(input[0].Height * (options.Scale ?? Defaults.Scale));
            output = (width, height);
            var update = new VideoEnhancementUpdate(VideoEnhancementPhase.Encoding, batch, batches)
            {
                FirstFrame = finished,
                FinishedFrames = finished,
                TotalFrames = known,
                Seed = seed,
                FramesPerSecond = rate,
                Size = output,
            };
            yield return update;
            await Task.Yield();
            cancellationToken.ThrowIfCancellationRequested();
            yield return update with { Phase = VideoEnhancementPhase.Restoring };
            if (batch == 0 && Gate is { } gate)
            {
                await gate.Task.WaitAsync(cancellationToken);
            }

            var decoded = Enumerable.Range(0, input.Count).Select(i => FakeImageGenerator.Gradient(width, height, (int)(finished + i) * 10)).ToList();
            finished += decoded.Count;
            yield return update with { Phase = VideoEnhancementPhase.Decoding, Frames = decoded, FinishedFrames = finished };
            if (AfterBatch is { } after)
            {
                await after(batch).WaitAsync(cancellationToken);
            }

            batch++;
        }

        yield return new VideoEnhancementUpdate(VideoEnhancementPhase.Completed, batch - 1, batch)
        {
            FirstFrame = finished,
            FinishedFrames = finished,
            TotalFrames = finished,
            Seed = seed,
            FramesPerSecond = rate,
            Size = output,
        };
    }

    public object? GetService(Type serviceType, object? serviceKey = null) => null;

    public void Dispose()
    {
    }
}
