using System.Diagnostics;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// The threading contract of the image generator on the tiny pipeline: calls return before compiling or running a graph, the streaming
/// loop runs on a background thread (the consumer's MoveNextAsync never runs a step), and cancelling stops the run between steps.
/// </summary>
public sealed class ThreadingTests
{
    [Fact]
    public async Task GenerateAsync_ReturnsBeforeTheFirstGraphRuns()
    {
        Assert.SkipUnless(Directory.Exists(Env.Tiny), "tiny-random-sd missing");
        using var model = Env.Load(Env.TinyDiffusers, Env.Cpu());
        using var generator = Env.Clients.Create<IImageGenerator>(model);
        var request = new ImageGenerationRequest { Prompt = "a red cube" };
        var options = new ImageGenerationOptions { Steps = 4, Seed = 3 };

        var watch = Stopwatch.StartNew();
        var running = generator.GenerateAsync(request, options, TestContext.Current.CancellationToken);
        var returned = watch.Elapsed;
        Assert.False(running.IsCompleted, "the generation ran before the call returned");
        var result = await running;
        TestContext.Current.SendDiagnosticMessage($"returned after {returned.TotalMilliseconds:F2} ms, done after {watch.Elapsed.TotalMilliseconds:F0} ms");
        Assert.True(returned < watch.Elapsed / 4, $"returned after {returned.TotalMilliseconds:F1} ms of {watch.Elapsed.TotalMilliseconds:F0} ms");
        Assert.Single(result.Images);
    }

    [Fact]
    public async Task Streaming_RunsStepsOffTheConsumersThread_AndCancelsBetweenSteps()
    {
        Assert.SkipUnless(Directory.Exists(Env.Tiny), "tiny-random-sd missing");
        using var model = Env.Load(Env.TinyDiffusers, Env.Cpu());
        using var generator = Env.Clients.Create<IImageGenerator>(model);
        var request = new ImageGenerationRequest { Prompt = "a red cube" };
        var options = new ImageGenerationOptions { Steps = 6, Seed = 3, PreviewInterval = 1 };

        // The time each MoveNextAsync call holds its caller, against the time the run takes: the steps (and the compilation on first use)
        // run on the background loop, so a call only hands over what the loop produced.
        var held = TimeSpan.Zero;
        var watch = Stopwatch.StartNew();
        var updates = new List<ImageGenerationUpdate>();
        await using (var stream = generator.GenerateStreamingAsync(request, options, TestContext.Current.CancellationToken).GetAsyncEnumerator(TestContext.Current.CancellationToken))
        {
            while (true)
            {
                var call = Stopwatch.StartNew();
                var next = stream.MoveNextAsync();
                held = held > call.Elapsed ? held : call.Elapsed;
                if (!await next)
                {
                    break;
                }

                updates.Add(stream.Current);
            }
        }

        TestContext.Current.SendDiagnosticMessage($"a MoveNextAsync call held its caller at most {held.TotalMilliseconds:F2} ms of {watch.Elapsed.TotalMilliseconds:F0} ms");
        Assert.Equal(updates[^1].TotalSteps + 1, updates.Count);
        Assert.NotNull(updates[^1].Result);
        Assert.True(held < watch.Elapsed / 4, $"a MoveNextAsync call held its caller {held.TotalMilliseconds:F1} ms");

        using var cancellation = new CancellationTokenSource();
        var seen = new List<ImageGenerationUpdate>();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (var update in generator.GenerateStreamingAsync(request, options with { Steps = 50 }, cancellation.Token))
            {
                seen.Add(update);
                if (update.Step == 2)
                {
                    cancellation.Cancel();
                }
            }
        });
        Assert.InRange(seen.Count, 2, 49);
        Assert.All(seen, update => Assert.Null(update.Result));

        // The client is idle again after the cancelled run: the next one starts at once and completes.
        var after = await generator.GenerateAsync(request, options with { PreviewInterval = 0 }, TestContext.Current.CancellationToken);
        Assert.Single(after.Images);
    }
}
