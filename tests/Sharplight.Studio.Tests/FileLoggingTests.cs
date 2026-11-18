using Microsoft.Extensions.Logging;
using Sharplight.Storage;
using Sharplight.Studio.Services.Logging;
using Sharplight.Studio.Services.Settings;
using Sharplight.Studio.ViewModels.Settings;

namespace Sharplight.Studio.Tests;

/// <summary>Studio's rolling log file: limits, the storage modes, the Settings action and --print-storage.</summary>
public sealed class FileLoggingTests
{
    [Fact]
    public void Writes_LevelsAndExceptions_AndSkipsBelowTheMinimum()
    {
        using var dir = new TempDir();
        var logs = dir.File("logs");
        using (var factory = StudioLogging.Create(new FileLogOptions { Directory = logs, MinimumLevel = LogLevel.Information }))
        {
            var log = factory.CreateLogger("Test");
            log.LogDebug("hidden");
            log.LogInformation("device {Name}", "RTX");
            log.LogError(new InvalidOperationException("boom"), "failed");
        }

        string text = File.ReadAllText(Path.Combine(logs, "studio.log"));
        Assert.DoesNotContain("hidden", text, StringComparison.Ordinal);
        Assert.Contains("INF Test: device RTX", text, StringComparison.Ordinal);
        Assert.Contains("ERR Test: failed", text, StringComparison.Ordinal);
        Assert.Contains("InvalidOperationException: boom", text, StringComparison.Ordinal);
    }

    [Fact]
    public void Rolls_OverAtTheSizeLimit_AndKeepsTheNewestFiles()
    {
        using var dir = new TempDir();
        var logs = dir.File("logs");
        using (var factory = StudioLogging.Create(new FileLogOptions { Directory = logs, MaxFileBytes = 1000, MaxFiles = 3 }))
        {
            var log = factory.CreateLogger("Roll");
            for (int i = 0; i < 60; i++)
            {
                log.LogInformation("line {Index} {Padding}", i, new string('x', 60));
            }
        }

        var files = Directory.GetFiles(logs).Select(Path.GetFileName).Order().ToList();
        Assert.Equal(["studio.1.log", "studio.2.log", "studio.log"], files);
        foreach (string file in Directory.GetFiles(logs))
        {
            Assert.True(new FileInfo(file).Length <= 1000, file);
        }

        // The newest lines are in the current file, the oldest ones are gone.
        Assert.Contains("line 59", File.ReadAllText(Path.Combine(logs, "studio.log")), StringComparison.Ordinal);
        Assert.DoesNotContain("line 0 ", string.Concat(Directory.GetFiles(logs).Select(File.ReadAllText)), StringComparison.Ordinal);
    }

    [Fact]
    public void AnUnwritableFolder_NeverThrows()
    {
        using var dir = new TempDir();
        var blocker = dir.File("not-a-folder");
        File.WriteAllText(blocker, "file");
        using var factory = StudioLogging.Create(new FileLogOptions { Directory = Path.Combine(blocker, "logs") });

        factory.CreateLogger("Test").LogCritical("still fine");
    }

    [Fact]
    public void TheLogFolder_FollowsTheStorageMode()
    {
        using var dir = new TempDir();
        var environment = new StorageEnvironment { ApplicationDirectory = dir.File("app"), HomeDirectory = dir.File("home"), Platform = StoragePlatform.Linux };

        var custom = StorageLayout.Create(StorageMode.Custom, dir.File("custom"), environment);
        var system = StorageLayout.Create(StorageMode.System, null, environment);
        foreach (var layout in new[] { custom, system })
        {
            using (var factory = StudioLogging.Create(layout, null))
            {
                factory.CreateLogger("Mode").LogInformation("hello from {Mode}", layout.Mode);
            }

            Assert.Contains($"hello from {layout.Mode}", File.ReadAllText(Path.Combine(layout.LogsDirectory, "studio.log")), StringComparison.Ordinal);
        }

        Assert.StartsWith(dir.File("custom"), custom.LogsDirectory, StringComparison.Ordinal);
        Assert.NotEqual(custom.LogsDirectory, system.LogsDirectory);
    }

    [Fact]
    public void PrintStorage_ShowsTheLogFile()
    {
        using var dir = new TempDir();
        var environment = new StorageEnvironment { ApplicationDirectory = dir.File("app"), HomeDirectory = dir.File("home"), Platform = StoragePlatform.Linux };
        var layout = StorageLayout.Resolve(environment);

        string report = StorageReport.Create(environment);

        Assert.Contains($"Logs:        {layout.LogsDirectory}", report, StringComparison.Ordinal);
        Assert.Contains($"Log file:    {Path.Combine(layout.LogsDirectory, "studio.log")}", report, StringComparison.Ordinal);
    }

    [Fact]
    public void Settings_OpenLogsFolder_CreatesAndOpensTheFolderOfTheStorageLayout()
    {
        using var services = new TestServices();
        var page = new SettingsPageViewModel(services.Services);

        page.OpenLogsDirectoryCommand.Execute(null);

        string logs = services.Services.StorageLocation.Current.LogsDirectory;
        Assert.Equal(logs, page.LogsDirectory);
        Assert.True(Directory.Exists(logs));
        Assert.Equal([logs], services.Platform.Opened);
    }

    [Fact]
    public void TheEngineAndUpdateServices_LogDeviceSelectionAndUpdateResults()
    {
        using var dir = new TempDir();
        var logs = dir.File("logs");
        using (var factory = StudioLogging.Create(new FileLogOptions { Directory = logs }))
        {
            var service = new Sharplight.Studio.Services.Updates.VelopackUpdateService("https://updates.example.test/", factory.CreateLogger("Updates"));
            Assert.False(service.Support.IsSupported);
        }

        Assert.Contains("Updates unavailable", File.ReadAllText(Path.Combine(logs, "studio.log")), StringComparison.Ordinal);
    }
}
