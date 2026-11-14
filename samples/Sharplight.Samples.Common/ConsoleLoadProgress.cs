using System.Diagnostics;
using Sharplight.Runtime;

namespace Sharplight.Samples;

/// <summary>
/// Shows model-load progress on stderr: one updating line when stderr is a terminal, otherwise a plain line when compiling,
/// uploading or a whole load finishes.
/// Use <see cref="Log"/> as the load's log callback so log lines do not mix with the progress line.
/// </summary>
public sealed class ConsoleLoadProgress : IProgress<LoadProgress>
{
    private static readonly long s_interval = Stopwatch.Frequency / 10;
    private readonly Lock _gate = new();
    private readonly string _prefix;
    private readonly bool _live;
    private long _lastDraw;
    private readonly SortedDictionary<LoadStage, LoadProgress> _active = [];
    private int _lineLength;

    /// <summary>Creates the display.</summary>
    /// <param name="prefix">Text put before every line (e.g. the model name), or empty.</param>
    /// <param name="live">Whether to draw an updating line; null = when stderr is a terminal.</param>
    public ConsoleLoadProgress(string prefix = "", bool? live = null)
    {
        _prefix = prefix;
        _live = live ?? !Console.IsErrorRedirected;
    }

    /// <summary>Writes a log line to stderr, clearing the progress line first.</summary>
    /// <param name="message">The message.</param>
    public void Log(string message)
    {
        lock (_gate)
        {
            ClearLine();
            Console.Error.WriteLine(_prefix + message);
        }
    }

    /// <inheritdoc/>
    public void Report(LoadProgress value)
    {
        lock (_gate)
        {
            if (!_live)
            {
                if (value.IsComplete && value.Stage is LoadStage.Compiling or LoadStage.Uploading or LoadStage.Ready)
                {
                    Console.Error.WriteLine(_prefix + Describe(value));
                }

                return;
            }

            // Stages may run concurrently (kernels compile while weights upload): the line shows every unfinished one.
            if (value.IsComplete)
            {
                _active.Remove(value.Stage);
            }
            else
            {
                _active[value.Stage] = value;
            }

            long now = Stopwatch.GetTimestamp();
            if (value.Stage != LoadStage.Ready && _active.Count > 0 && now - _lastDraw < s_interval)
            {
                return;
            }

            _lastDraw = now;
            ClearLine();
            if (value.Stage == LoadStage.Ready)
            {
                Console.Error.WriteLine(_prefix + Describe(value));
                return;
            }

            IEnumerable<LoadProgress> shown = _active.Count > 0 ? _active.Values : [value];
            string text = _prefix + string.Join(" | ", shown.Select(Describe));
            Console.Error.Write(text);
            _lineLength = text.Length;
        }
    }

    private void ClearLine()
    {
        if (_lineLength > 0)
        {
            Console.Error.Write("\r" + new string(' ', _lineLength) + "\r");
            _lineLength = 0;
        }
    }

    private static string Describe(LoadProgress p)
    {
        string detail = p.Detail is { } d ? $" ({d})" : "";
        return p.Stage switch
        {
            LoadStage.Ready => $"Ready{detail}",
            LoadStage.Uploading => $"Uploading weights {p.Completed / 1e9:F2} / {p.Total / 1e9:F2} GB{detail}",
            LoadStage.Compiling => $"Compiling {p.Completed} / {p.Total}{(p.Cached > 0 ? $", {p.Cached} cached" : "")}{detail}",
            _ => $"{p.Stage} {p.Fraction * 100:F0}%{detail}",
        };
    }
}
