using System.Runtime.CompilerServices;
using Sharplight.Tasks;
using Xunit;

namespace Sharplight.Core.Tests.Tasks;

/// <summary>
/// The threading contract of the task clients (<see cref="BackgroundCompute"/>): calls return before their compute is done, the compute
/// never runs on the caller's thread, streams are produced without waiting for the consumer, and cancellation stops between steps after
/// the step in progress.
/// </summary>
public class BackgroundComputeTests
{
    [Fact]
    public async Task RunAsync_ReturnsBeforeTheWorkIsDone_AndRunsItOnAnotherThread()
    {
        var release = new ManualResetEventSlim();
        int caller = Environment.CurrentManagedThreadId;
        int worker = caller;
        var task = BackgroundCompute.RunAsync(() =>
        {
            worker = Environment.CurrentManagedThreadId;
            release.Wait(TestContext.Current.CancellationToken);
            return 42;
        }, TestContext.Current.CancellationToken);

        Assert.False(task.IsCompleted);
        release.Set();
        Assert.Equal(42, await task);
        Assert.NotEqual(caller, worker);
    }

    [Fact]
    public async Task RunAsync_WithAGate_RunsOneCallAtATime()
    {
        using var gate = new SemaphoreSlim(1, 1);
        int running = 0;
        int most = 0;
        var calls = Enumerable.Range(0, 4).Select(_ => BackgroundCompute.RunAsync(gate, () =>
        {
            int now = Interlocked.Increment(ref running);
            InterlockedMax(ref most, now);
            Thread.Sleep(20);
            Interlocked.Decrement(ref running);
            return now;
        }, TestContext.Current.CancellationToken)).ToList();

        await Task.WhenAll(calls);
        Assert.Equal(1, most);
    }

    [Fact]
    public async Task StreamAsync_ProducesOffTheConsumersThread_WithoutWaitingForTheConsumer()
    {
        var produced = new TaskCompletionSource();
        var inline = new List<bool>();
        var stream = BackgroundCompute.StreamAsync(token => Produce(5, inline, produced, token), TestContext.Current.CancellationToken);
        await using var items = stream.GetAsyncEnumerator(TestContext.Current.CancellationToken);

        // The first MoveNextAsync starts the producer; the consumer then stops reading, yet every item gets produced.
        s_insideCall = true;
        var first = items.MoveNextAsync();
        s_insideCall = false;
        Assert.True(await first);
        await produced.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        var all = new List<int> { items.Current };
        while (true)
        {
            s_insideCall = true;
            var next = items.MoveNextAsync();
            s_insideCall = false;
            if (!await next)
            {
                break;
            }

            all.Add(items.Current);
        }

        Assert.Equal([0, 1, 2, 3, 4], all);
        lock (inline)
        {
            Assert.Equal(5, inline.Count);
            Assert.DoesNotContain(true, inline);
        }
    }

    [Fact]
    public async Task StreamAsync_SurfacesTheProducersExceptionAfterItsItems()
    {
        var items = new List<int>();
        var error = await Assert.ThrowsAsync<InvalidDataException>(async () =>
        {
            await foreach (int item in BackgroundCompute.StreamAsync(Failing, TestContext.Current.CancellationToken))
            {
                items.Add(item);
            }
        });

        Assert.Equal("broken", error.Message);
        Assert.Equal([1, 2], items);
    }

    [Fact]
    public async Task StreamAsync_Cancelled_StopsTheConsumerAtOnce_AndEndsAfterTheStepInProgress()
    {
        using var cancellation = new CancellationTokenSource();
        var stepStarted = new TaskCompletionSource();
        var finishStep = new ManualResetEventSlim();
        bool stepFinished = false;
        int steps = 0;

        async IAsyncEnumerable<int> Slow([EnumeratorCancellation] CancellationToken token)
        {
            for (int i = 0; ; i++)
            {
                token.ThrowIfCancellationRequested();
                Interlocked.Increment(ref steps);
                await Task.Run(() =>
                {
                    stepStarted.TrySetResult();
                    finishStep.Wait(TestContext.Current.CancellationToken);
                    stepFinished = true;
                }, CancellationToken.None);
                yield return i;
            }
        }

        var consume = Task.Run(async () =>
        {
            await foreach (int _ in BackgroundCompute.StreamAsync(Slow, cancellation.Token))
            {
            }
        }, TestContext.Current.CancellationToken);

        await stepStarted.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        cancellation.Cancel();

        // The enumeration does not end while the step runs (the client is still in use), and ends right after it.
        await Task.Delay(100, TestContext.Current.CancellationToken);
        Assert.False(consume.IsCompleted);
        finishStep.Set();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => consume.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken));
        Assert.True(stepFinished);
        Assert.Equal(1, steps);
    }

    [Fact]
    public async Task StreamAsync_LeftEarly_StopsTheProducer()
    {
        int produced = 0;

        async IAsyncEnumerable<int> Endless([EnumeratorCancellation] CancellationToken token)
        {
            for (int i = 0; ; i++)
            {
                await Task.Delay(5, token);
                Interlocked.Increment(ref produced);
                yield return i;
            }
        }

        await foreach (int item in BackgroundCompute.StreamAsync(Endless, TestContext.Current.CancellationToken))
        {
            if (item == 2)
            {
                break;
            }
        }

        int after = Volatile.Read(ref produced);
        await Task.Delay(100, TestContext.Current.CancellationToken);
        Assert.Equal(after, Volatile.Read(ref produced));
    }

    /// <summary>Whether the current thread is inside a MoveNextAsync call of the consumer (a producer step there ran inline).</summary>
    [Fact]
    public async Task StreamBlocking_WithACapacity_HoldsTheProducerBackBehindASlowConsumer()
    {
        const int capacity = 3;
        int produced = 0;
        int consumed = 0;
        int mostAhead = 0;
        var stream = BackgroundCompute.StreamBlocking<int>((emit, token) =>
        {
            for (int i = 0; i < 40; i++)
            {
                token.ThrowIfCancellationRequested();
                emit(i);
                InterlockedMax(ref mostAhead, Interlocked.Increment(ref produced) - Volatile.Read(ref consumed));
            }
        }, capacity, TestContext.Current.CancellationToken);

        var items = new List<int>();
        await foreach (int item in stream)
        {
            await Task.Delay(5, TestContext.Current.CancellationToken);
            items.Add(item);
            Interlocked.Increment(ref consumed);
        }

        Assert.Equal(Enumerable.Range(0, 40), items);
        // The queue, the item the consumer holds and the one being emitted.
        Assert.True(mostAhead <= capacity + 2, $"the producer ran {mostAhead} items ahead");
    }

    [Fact]
    public async Task StreamBlocking_WithACapacity_StopsABlockedProducerWhenTheConsumerLeaves()
    {
        var stopped = new TaskCompletionSource();
        var stream = BackgroundCompute.StreamBlocking<int>((emit, token) =>
        {
            try
            {
                for (int i = 0; ; i++)
                {
                    emit(i);
                }
            }
            finally
            {
                stopped.TrySetResult();
            }
        }, 1, TestContext.Current.CancellationToken);

        await foreach (int item in stream)
        {
            if (item == 2)
            {
                break;
            }
        }

        await stopped.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
    }

    [ThreadStatic]
    private static bool s_insideCall;

    private static async IAsyncEnumerable<int> Produce(int count, List<bool> inline, TaskCompletionSource produced, [EnumeratorCancellation] CancellationToken token)
    {
        for (int i = 0; i < count; i++)
        {
            // A blocking step, as a graph run is: it would hold up the consumer if it ran inside its MoveNextAsync call.
            Thread.Sleep(5);
            token.ThrowIfCancellationRequested();
            lock (inline)
            {
                inline.Add(s_insideCall);
            }

            yield return i;
            await Task.Yield();
        }

        produced.SetResult();
    }

    private static async IAsyncEnumerable<int> Failing([EnumeratorCancellation] CancellationToken token)
    {
        await Task.Yield();
        yield return 1;
        yield return 2;
        token.ThrowIfCancellationRequested();
        throw new InvalidDataException("broken");
    }

    private static void InterlockedMax(ref int target, int value)
    {
        int current = Volatile.Read(ref target);
        while (value > current)
        {
            int seen = Interlocked.CompareExchange(ref target, value, current);
            if (seen == current)
            {
                return;
            }

            current = seen;
        }
    }
}
