using System.Runtime.CompilerServices;
using System.Runtime.ExceptionServices;
using System.Threading.Channels;

namespace Sharplight.Tasks;

/// <summary>
/// Runs the compute of task clients off the caller's thread (the threading contract of the task layer, docs/api-overview.md): an
/// <c>...Async</c> call returns to its caller before any graph is compiled or run, the work runs on a dedicated thread (graph runs block
/// on device fences for a long time, which would starve the thread pool), and streamed results are produced by a background loop
/// that never waits for the consumer. Cancellation is honored between steps; a cancelled or abandoned stream waits for the step in
/// progress before it completes, so the client (and the model behind it) is no longer in use once the enumeration has ended.
/// </summary>
internal static class BackgroundCompute
{
    /// <summary>Runs <paramref name="work"/> on a dedicated thread.</summary>
    /// <typeparam name="T">The result type.</typeparam>
    /// <param name="work">The blocking compute.</param>
    /// <param name="cancellationToken">Cancels the work if it has not started yet.</param>
    /// <returns>The result of <paramref name="work"/>.</returns>
    public static Task<T> RunAsync<T>(Func<T> work, CancellationToken cancellationToken)
        => Task.Factory.StartNew(work, cancellationToken, TaskCreationOptions.LongRunning | TaskCreationOptions.DenyChildAttach, TaskScheduler.Default);

    /// <summary>Runs <paramref name="work"/> on a dedicated thread, one call at a time per <paramref name="gate"/>.</summary>
    /// <typeparam name="T">The result type.</typeparam>
    /// <param name="gate">Serializes the calls of one client (its compiled graphs hold the inputs and outputs of one run).</param>
    /// <param name="work">The blocking compute.</param>
    /// <param name="cancellationToken">Cancels the wait for the gate and the work if it has not started yet.</param>
    /// <returns>The result of <paramref name="work"/>.</returns>
    public static async Task<T> RunAsync<T>(SemaphoreSlim gate, Func<T> work, CancellationToken cancellationToken)
    {
        await gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            return await RunAsync(work, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            gate.Release();
        }
    }

    /// <summary>
    /// Enumerates <paramref name="source"/> on a background task and hands its items to the caller through an unbounded channel, so the
    /// producing loop never runs on the consumer's thread and never waits for a slow consumer. Exceptions of the source surface at the
    /// consumer after the items produced before them.
    /// </summary>
    /// <typeparam name="T">The item type.</typeparam>
    /// <param name="source">Creates the producing sequence; its token is cancelled when the consumer cancels or stops early.</param>
    /// <param name="cancellationToken">Stops the consumer at once and the producer at its next step.</param>
    /// <returns>The items of <paramref name="source"/>, in order.</returns>
    public static async IAsyncEnumerable<T> StreamAsync<T>(Func<CancellationToken, IAsyncEnumerable<T>> source,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var channel = Channel.CreateUnbounded<T>(new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });
        ExceptionDispatchInfo? failure = null;
        var producer = Task.Run(async () =>
        {
            try
            {
                await foreach (var item in source(stop.Token).ConfigureAwait(false))
                {
                    channel.Writer.TryWrite(item);
                }
            }
            catch (Exception error)
            {
                failure = ExceptionDispatchInfo.Capture(error);
            }
            finally
            {
                channel.Writer.TryComplete();
            }
        }, CancellationToken.None);

        try
        {
            while (await channel.Reader.WaitToReadAsync(cancellationToken).ConfigureAwait(false))
            {
                while (channel.Reader.TryRead(out var item))
                {
                    yield return item;
                }
            }

            failure?.Throw();
        }
        finally
        {
            stop.Cancel();
            await producer.ConfigureAwait(false);
        }
    }
}
