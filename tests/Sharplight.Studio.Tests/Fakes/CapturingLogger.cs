using Microsoft.Extensions.Logging;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>A logger that keeps what it was given (the message and the exception text), for tests of what the log says.</summary>
internal sealed class CapturingLogger : ILogger
{
    private readonly List<string> _lines = [];

    /// <summary>The lines logged so far.</summary>
    public IReadOnlyList<string> Lines
    {
        get
        {
            lock (_lines)
            {
                return [.. _lines];
            }
        }
    }

    public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

    public bool IsEnabled(LogLevel logLevel) => true;

    public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
    {
        lock (_lines)
        {
            _lines.Add($"{logLevel}: {formatter(state, exception)}" + (exception is null ? "" : " | " + exception.Message));
        }
    }
}
