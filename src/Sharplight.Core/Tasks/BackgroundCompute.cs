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
    public static IAsyncEnumerable<T> StreamAsync<T>(Func<CancellationToken, IAsyncEnumerable<T>> source, CancellationToken cancellationToken)
        => Pump<T>(async (emit, token) =>
        {
            await foreach (var item in source(token).ConfigureAwait(false))
            {
                emit(item);
            }
        }, null, cancellationToken);

    /// <summary>
    /// Runs the blocking <paramref name="produce"/> on a dedicated thread and hands every item it emits to the caller as soon as it is emitted
    /// (a producer that emits from inside its compute loop, such as audio chunks while the model is still generating). Exceptions surface at the
    /// consumer after the items emitted before them.
    /// </summary>
    /// <typeparam name="T">The item type.</typeparam>
    /// <param name="produce">The blocking compute; calls its first argument for each item, honors its token between steps.</param>
    /// <param name="cancellationToken">Stops the consumer at once and the producer at its next step.</param>
    /// <returns>The emitted items, in order.</returns>
    public static IAsyncEnumerable<T> StreamBlocking<T>(Action<Action<T>, CancellationToken> produce, CancellationToken cancellationToken)
        => Pump<T>((emit, token) => Task.Factory.StartNew(() => produce(emit, token), CancellationToken.None, TaskCreationOptions.LongRunning | TaskCreationOptions.DenyChildAttach,
            TaskScheduler.Default), null, cancellationToken);

    /// <summary>
    /// Like <see cref="StreamBlocking{T}(Action{Action{T}, CancellationToken}, CancellationToken)"/>, but at most <paramref name="capacity"/>
    /// items wait for the consumer: an emit beyond that blocks the producing thread (between its steps) until the consumer takes one, so a
    /// producer of large items (frames) cannot run ahead of a slow consumer and pile them up.
    /// </summary>
    /// <typeparam name="T">The item type.</typeparam>
    /// <param name="produce">The blocking compute; calls its first argument for each item, honors its token between steps.</param>
    /// <param name="capacity">The most items queued for the consumer (positive).</param>
    /// <param name="cancellationToken">Stops the consumer at once and the producer at its next step (or its blocked emit).</param>
    /// <returns>The emitted items, in order.</returns>
    public static IAsyncEnumerable<T> StreamBlocking<T>(Action<Action<T>, CancellationToken> produce, int capacity, CancellationToken cancellationToken)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
        return Pump<T>((emit, token) => Task.Factory.StartNew(() => produce(emit, token), CancellationToken.None, TaskCreationOptions.LongRunning | TaskCreationOptions.DenyChildAttach,
            TaskScheduler.Default), capacity, cancellationToken);
    }

    private static async IAsyncEnumerable<T> Pump<T>(Func<Action<T>, CancellationToken, Task> produce, int? capacity, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var channel = capacity is { } bound
            ? Channel.CreateBounded<T>(new BoundedChannelOptions(bound) { SingleReader = true, SingleWriter = true, FullMode = BoundedChannelFullMode.Wait })
            : Channel.CreateUnbounded<T>(new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });
        ExceptionDispatchInfo? failure = null;
        var producer = Task.Run(async () =>
        {
            try
            {
                await produce(item => Write(channel.Writer, item, stop.Token), stop.Token).ConfigureAwait(false);
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

    /// <summary>Queues an item; waits (blocking the producing thread) while a bounded channel is full.</summary>
    private static void Write<T>(ChannelWriter<T> writer, T item, CancellationToken cancellationToken)
    {
        while (!writer.TryWrite(item))
        {
            if (!writer.WaitToWriteAsync(cancellationToken).AsTask().GetAwaiter().GetResult())
            {
                return;
            }
        }
    }
}
