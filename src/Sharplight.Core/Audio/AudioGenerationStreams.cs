using System.Runtime.CompilerServices;
using Sharplight.Tasks;

namespace Sharplight.Audio;

/// <summary>The streaming and whole-result forms of the audio generators: a gated blocking producer of <see cref="AudioGenerationUpdate"/>s.</summary>
internal static class AudioGenerationStreams
{
    /// <summary>
    /// Runs <paramref name="produce"/> on a dedicated thread, one request at a time per <paramref name="gate"/>, and hands every update it emits to the caller as
    /// soon as it is emitted; the producer emits the last update with the result.
    /// </summary>
    public static async IAsyncEnumerable<AudioGenerationUpdate> Stream(SemaphoreSlim gate, Action<Action<AudioGenerationUpdate>, CancellationToken> produce,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        await gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            await foreach (var update in BackgroundCompute.StreamBlocking(produce, cancellationToken).ConfigureAwait(false))
            {
                yield return update;
            }
        }
        finally
        {
            gate.Release();
        }
    }

    /// <summary>The result of the last update of <paramref name="updates"/>.</summary>
    public static async Task<AudioGenerationResult> CompleteAsync(IAsyncEnumerable<AudioGenerationUpdate> updates)
    {
        AudioGenerationResult? result = null;
        await foreach (var update in updates.ConfigureAwait(false))
        {
            result = update.Result ?? result;
        }

        return result ?? throw new InvalidOperationException("The generation ended without a result.");
    }

    /// <summary>The seed of a request: the requested one, or a new one (53 bits, so it survives a round trip through a double).</summary>
    public static long ResolveSeed(long? requested) => requested ?? Random.Shared.NextInt64(1L << 53);
}
