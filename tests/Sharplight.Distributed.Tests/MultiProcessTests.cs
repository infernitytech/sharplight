using System.Diagnostics;

namespace Sharplight.Distributed.Tests;

/// <summary>
/// Segment workers as separate processes (<c>sharplight segment-worker</c> of the CLI build) on loopback: the pipeline over them generates the
/// tokens of a single CPU run, and killing one fails the next step promptly. Skipped when the CLI is not built or the model is missing.
/// </summary>
[Collection("Workers")]
public class MultiProcessTests
{
    private const string Prompt = "The capital of France is";

    [Fact]
    public void WorkerProcessesGenerateTheSameTokensAndFailCleanlyWhenKilled()
    {
        string path = TestModels.Require(TestModels.SmolLm);
        string cli = CliPath();
        int[] expected;
        using (var cpu = TestModels.Cpu())
        using (var single = TestModels.Load(path, [cpu]))
        using (var decoder = single.CreateBatchDecoder(1, 256))
        {
            expected = TestModels.Greedy(decoder, single, Prompt, 8).Tokens;
        }

        using var first = WorkerProcess.Start(cli, path);
        using var second = WorkerProcess.Start(cli, path);
        using var provider = new RemoteExecutionProvider(new RemoteSegmentOptions
        {
            Workers = [first.Endpoint, second.Endpoint],
            HeartbeatInterval = TimeSpan.FromMilliseconds(200),
            HeartbeatTimeout = TimeSpan.FromSeconds(3),
        });
        var remote = provider.OpenAll();
        try
        {
            using var local = TestModels.Cpu();
            using var split = TestModels.Load(path, [local, .. remote]);
            using var decoder = split.CreateBatchDecoder(1, 256);
            Assert.Equal(expected, TestModels.Greedy(decoder, split, Prompt, 8).Tokens);

            second.Kill();
            var watch = Stopwatch.StartNew();
            Assert.Throws<SegmentWorkerException>(() => TestModels.Greedy(decoder, split, Prompt, 4));
            Assert.True(watch.Elapsed < TimeSpan.FromSeconds(10), $"the failure took {watch.Elapsed}");
        }
        finally
        {
            foreach (var device in remote)
            {
                device.Dispose();
            }
        }
    }

    /// <summary>The built CLI (Release, then Debug) next to this repository's tests, or skips the test.</summary>
    private static string CliPath()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "Sharplight.slnx")))
        {
            directory = directory.Parent;
        }

        var candidates = directory is null ? [] : new[] { "Release", "Debug" }
            .Select(c => Path.Combine(directory.FullName, "samples/Sharplight.Cli/bin", c, "net11.0", OperatingSystem.IsWindows() ? "sharplight.exe" : "sharplight"));
        var cli = candidates.FirstOrDefault(File.Exists);
        Assert.SkipUnless(cli is not null, "the CLI is not built (dotnet build samples/Sharplight.Cli -c Release)");
        return cli!;
    }

    /// <summary>A <c>sharplight segment-worker</c> process on a free loopback port, killed on dispose.</summary>
    private sealed class WorkerProcess : IDisposable
    {
        private readonly Process _process;

        private WorkerProcess(Process process, string endpoint)
        {
            _process = process;
            Endpoint = endpoint;
        }

        public string Endpoint { get; }

        public static WorkerProcess Start(string cli, string model)
        {
            var start = new ProcessStartInfo(cli, ["segment-worker", model, "--cpu", "--listen", "127.0.0.1:0"])
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            };
            start.Environment["CUDA_VISIBLE_DEVICES"] = "";
            start.Environment["SHARPLIGHT_VK_DEVICES"] = "99";
            start.Environment["SHARPLIGHT_CPU_THREADS"] = "2";
            Process process;
            try
            {
                process = Process.Start(start) ?? throw new InvalidOperationException("no process");
            }
            catch (Exception e) when (e is System.ComponentModel.Win32Exception or InvalidOperationException)
            {
                Assert.Skip($"cannot start the CLI: {e.Message}");
                throw;
            }

            process.ErrorDataReceived += (_, _) => { };
            process.BeginErrorReadLine();
            var line = process.StandardOutput.ReadLineAsync();
            if (!line.Wait(TimeSpan.FromSeconds(60)) || line.Result is not { } text || !text.StartsWith("listening on ", StringComparison.Ordinal))
            {
                process.Kill();
                Assert.Skip("the segment worker process did not start");
            }

            return new WorkerProcess(process, line.Result!["listening on ".Length..].Trim());
        }

        public void Kill()
        {
            _process.Kill();
            _process.WaitForExit();
        }

        public void Dispose()
        {
            if (!_process.HasExited)
            {
                Kill();
            }

            _process.Dispose();
        }
    }
}
