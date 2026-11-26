using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using Sharplight.Vision;

namespace Sharplight.Studio.ViewModels.Video;

/// <summary>
/// Keeps the input frames the enhancer has read and not yet finished, to pair them with the enhanced frames in the comparison: a tap on the
/// frame source records each frame, and each finished batch takes its frames out again. It holds what the enhancer holds (a few batches),
/// never the clip. Thread safe: the source is read on the enhancer's side, the frames are taken on the page's.
/// </summary>
internal sealed class OriginalFrameQueue
{
    private readonly ConcurrentQueue<ImageInput> _frames = new();

    /// <summary>The frames read and not taken yet.</summary>
    public int Count => _frames.Count;

    /// <summary>The frames of <paramref name="source"/>, each recorded as it passes.</summary>
    /// <param name="source">The input frames in order.</param>
    /// <param name="cancellationToken">Stops the reading.</param>
    public async IAsyncEnumerable<ImageInput> Tap(IAsyncEnumerable<ImageInput> source, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await foreach (var frame in source.WithCancellation(cancellationToken).ConfigureAwait(false))
        {
            _frames.Enqueue(frame);
            yield return frame;
        }
    }

    /// <summary>Takes the next <paramref name="count"/> recorded frames (the inputs of the batch that just finished), in order.</summary>
    /// <param name="count">The frames of the finished batch.</param>
    /// <returns>The frames; fewer when fewer were recorded.</returns>
    public List<ImageInput> Take(int count)
    {
        var frames = new List<ImageInput>(count);
        while (frames.Count < count && _frames.TryDequeue(out var frame))
        {
            frames.Add(frame);
        }

        return frames;
    }
}
