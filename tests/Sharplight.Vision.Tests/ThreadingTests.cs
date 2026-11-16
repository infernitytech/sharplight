using System.Diagnostics;
using Sharplight.Imaging;
using Sharplight.Tasks;

namespace Sharplight.Vision.Tests;

/// <summary>
/// The threading contract of the vision clients: a call returns before the graph has run (the run happens on a background thread), the
/// result is the same as before, and calls on one client do not interfere. Skipped when the checkpoints are not present.
/// </summary>
public class ThreadingTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    private static string Model(string name) => Path.Combine(Home, "models/vision", name);

    [Fact]
    public async Task ClassifyAsync_ReturnsBeforeTheGraphRuns_AndConcurrentCallsAgree()
    {
        Assert.SkipUnless(File.Exists(Path.Combine(Model("vit-base-patch16-224"), "sharplight-sample.png")), "vit-base-patch16-224 missing");
        var image = PngCodec.Read(Path.Combine(Model("vit-base-patch16-224"), "sharplight-sample.png"));
        using var model = TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = Model("vit-base-patch16-224"), Devices = [Env.Cpu()] });
        using var classifier = new TaskClientProvider([.. Env.TaskFactories()]).Create<IImageClassifier>(model);
        var expected = await classifier.ClassifyAsync(image, cancellationToken: TestContext.Current.CancellationToken);

        var watch = Stopwatch.StartNew();
        var first = classifier.ClassifyAsync(image, cancellationToken: TestContext.Current.CancellationToken);
        var second = classifier.ClassifyAsync(image, cancellationToken: TestContext.Current.CancellationToken);
        var returned = watch.Elapsed;
        Assert.False(first.IsCompleted && second.IsCompleted, "the calls ran their graphs before returning");
        var results = await Task.WhenAll(first, second);
        var total = watch.Elapsed;
        TestContext.Current.SendDiagnosticMessage($"returned after {returned.TotalMilliseconds:F2} ms, done after {total.TotalMilliseconds:F0} ms");
        Assert.True(returned < total / 4, $"returned after {returned.TotalMilliseconds:F1} ms of {total.TotalMilliseconds:F0} ms");
        foreach (var result in results)
        {
            Assert.Equal(expected.Predictions.Select(p => (p.Index, p.Logit)), result.Predictions.Select(p => (p.Index, p.Logit)));
        }
    }

    [Fact]
    public async Task EstimateAsync_ReturnsBeforeTheGraphRuns_AndHonorsCancellation()
    {
        Assert.SkipUnless(File.Exists(Path.Combine(Model("depth-anything-v2-small"), "sharplight-sample.png")), "depth-anything-v2-small missing");
        var image = PngCodec.Read(Path.Combine(Model("depth-anything-v2-small"), "sharplight-sample.png"));
        using var model = TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = Model("depth-anything-v2-small"), Devices = [Env.Cpu()] });
        using var estimator = new TaskClientProvider([.. Env.TaskFactories()]).Create<IDepthEstimator>(model);

        var watch = Stopwatch.StartNew();
        var running = estimator.EstimateAsync(image, cancellationToken: TestContext.Current.CancellationToken);
        var returned = watch.Elapsed;
        Assert.False(running.IsCompleted);
        var depth = await running;
        Assert.True(returned < watch.Elapsed / 4, $"returned after {returned.TotalMilliseconds:F1} ms of {watch.Elapsed.TotalMilliseconds:F0} ms");
        Assert.Equal((image.Width, image.Height), (depth.Width, depth.Height));

        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => estimator.EstimateAsync(image, cancellationToken: cancelled.Token));
    }
}
