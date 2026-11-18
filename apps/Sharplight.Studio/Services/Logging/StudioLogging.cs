using Microsoft.Extensions.Logging;
using Sharplight.Storage;

namespace Sharplight.Studio.Services.Logging;

/// <summary>Creates Studio's logger factory (a rolling file in the log folder of the storage layout) and hooks the process-wide error events.</summary>
internal static class StudioLogging
{
    /// <summary>The environment variable that sets the minimum level (<c>Trace</c> ... <c>Error</c>); default <c>Information</c>.</summary>
    public const string LevelVariable = "SHARPLIGHT_LOG_LEVEL";

    /// <summary>Creates the factory writing to <paramref name="options"/>.</summary>
    public static ILoggerFactory Create(FileLogOptions options) => LoggerFactory.Create(builder => builder.SetMinimumLevel(options.MinimumLevel)
        .AddProvider(new RollingFileLoggerProvider(options)));

    /// <summary>Creates the factory for the log folder of <paramref name="layout"/>.</summary>
    public static ILoggerFactory Create(StorageLayout layout, string? level)
        => Create(new FileLogOptions
        {
            Directory = layout.LogsDirectory,
            MinimumLevel = Enum.TryParse<LogLevel>(level, ignoreCase: true, out var parsed) ? parsed : LogLevel.Information,
        });

    /// <summary>Logs unhandled and unobserved exceptions of any thread.</summary>
    public static void Hook(ILogger logger)
    {
        AppDomain.CurrentDomain.UnhandledException += (_, args) =>
            logger.LogCritical(args.ExceptionObject as Exception, "Unhandled exception (terminating: {Terminating})", args.IsTerminating);
        TaskScheduler.UnobservedTaskException += (_, args) => logger.LogError(args.Exception, "Unobserved task exception");
    }
}
