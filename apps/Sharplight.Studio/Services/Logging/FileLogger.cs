using Microsoft.Extensions.Logging;

namespace Sharplight.Studio.Services.Logging;

/// <summary>A category's logger of <see cref="RollingFileLoggerProvider"/>.</summary>
internal sealed class FileLogger : ILogger
{
    private readonly RollingFileLoggerProvider _provider;
    private readonly string _category;
    private readonly LogLevel _minimum;

    public FileLogger(RollingFileLoggerProvider provider, string category, LogLevel minimum)
    {
        _provider = provider;
        _category = category;
        _minimum = minimum;
    }

    /// <inheritdoc/>
    public IDisposable? BeginScope<TState>(TState state)
        where TState : notnull => null;

    /// <inheritdoc/>
    public bool IsEnabled(LogLevel logLevel) => logLevel != LogLevel.None && logLevel >= _minimum;

    /// <inheritdoc/>
    public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
    {
        if (IsEnabled(logLevel))
        {
            _provider.Write(logLevel, _category, formatter(state, exception), exception);
        }
    }
}
