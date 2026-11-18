using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Sharplight.Audio;
using Sharplight.Storage;
using Sharplight.Studio.Services.Feed;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Services.Settings;
using Sharplight.Studio.Services.Updates;

namespace Sharplight.Studio.Services.Smoke;

/// <summary>
/// <c>sharplight-studio --smoke</c>: builds the real services as <see cref="StudioHost"/> does (settings, storage, updates, feed, the engine with all
/// providers and formats) without a window, runs <see cref="SmokeRunner"/>, writes the JSON report and returns the exit code (0 passed, 1 failed).
/// </summary>
internal static class SmokeCommand
{
    /// <summary>The longest the whole run may take.</summary>
    public static readonly TimeSpan Timeout = TimeSpan.FromMinutes(10);

    /// <summary>Runs the smoke test.</summary>
    /// <param name="layout">The storage layout in effect.</param>
    /// <param name="logs">The logger factory writing to the log folder.</param>
    /// <param name="reportPath">Where to write the JSON report, or null for <c>smoke-report.json</c> in the log folder.</param>
    public static int Run(StorageLayout layout, ILoggerFactory logs, string? reportPath)
    {
        var log = logs.CreateLogger("Sharplight.Studio.Smoke");
        SmokeReport report;
        try
        {
            report = RunServices(layout, logs, log);
        }
        catch (Exception error)
        {
            log.LogCritical(error, "The smoke run could not start the services");
            report = new SmokeReport
            {
                Version = Program.Version,
                Os = System.Runtime.InteropServices.RuntimeInformation.OSDescription,
                Architecture = System.Runtime.InteropServices.RuntimeInformation.OSArchitecture.ToString(),
                Storage = $"{layout.Mode} at {layout.Description}",
                Steps = [new SmokeStep("services", false, $"{error.GetType().Name}: {error.Message}", 0)],
            };
        }

        string json = JsonSerializer.Serialize(report, StudioJsonContext.Default.SmokeReport);
        string path = reportPath ?? Path.Combine(layout.LogsDirectory, "smoke-report.json");
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        File.WriteAllText(path, json);
        log.LogInformation("Smoke report written to {Path}: {Result}", path, report.Passed ? "passed" : "FAILED");
        Console.Out.WriteLine(json);
        logs.Dispose();
        return report.Passed ? 0 : 1;
    }

    private static SmokeReport RunServices(StorageLayout layout, ILoggerFactory logs, ILogger log)
    {
        var environment = StorageEnvironment.Process;
        StorageMigration.CompletePending(layout);
        var paths = StudioPaths.From(layout, environment.GetVariable(StudioPaths.HomeVariable));
        var settings = new JsonSettingsService(paths.SettingsFile, null, layout);
        using var container = new ServiceCollection().AddSingleton<ISettingsService>(settings).AddSingleton(logs)
            .AddSingleton(typeof(ILogger<>), typeof(Logger<>)).AddSingleton(paths).AddSharplightStorage(layout).AddStudioUpdates().AddStudioFeed()
            .AddSharplight().BuildServiceProvider();
        var engine = EngineServices.From(container);
        using var runtime = new EngineModelRuntime(engine, () => settings.Current.Residency, null, () => settings.Current.UseCompilationCache,
            logs.CreateLogger<EngineModelRuntime>());
        var runner = new SmokeRunner(container.GetRequiredService<IFeedService>(), runtime, container.GetRequiredService<IAudioDecoder>(),
            Path.Combine(paths.DataDirectory, "smoke"), log);
        using var cancel = new CancellationTokenSource(Timeout);
        return runner.RunAsync(Program.Version, $"{layout.Mode} at {layout.Description}", cancel.Token).GetAwaiter().GetResult();
    }
}
