using Avalonia.Logging;

namespace Sharplight.Studio.Tests;

/// <summary>Collects binding and property warnings Avalonia logs while views render.</summary>
internal sealed class BindingErrorSink : ILogSink
{
    private readonly Lock _lock = new();
    private readonly List<string> _messages = [];

    public IReadOnlyList<string> Messages
    {
        get
        {
            lock (_lock)
            {
                return [.. _messages];
            }
        }
    }

    public bool IsEnabled(LogEventLevel level, string area) => level >= LogEventLevel.Warning && area is LogArea.Binding or LogArea.Property;

    public void Log(LogEventLevel level, string area, object? source, string messageTemplate) => Add(area, source, messageTemplate, []);

    public void Log(LogEventLevel level, string area, object? source, string messageTemplate, params object?[] propertyValues) =>
        Add(area, source, messageTemplate, propertyValues);

    private void Add(string area, object? source, string template, object?[] values)
    {
        lock (_lock)
        {
            _messages.Add($"[{area}] {source?.GetType().Name}: {template} {string.Join(" | ", values)}");
        }
    }
}
