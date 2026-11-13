using System.Diagnostics;
using Sharplight.Hub;

namespace Sharplight.Samples.Hub;

/// <summary>
/// Shows pull progress on stderr: one line rewritten in place (bytes, percent, speed, ETA) on a terminal, or a plain line every
/// few seconds when stderr is redirected. Log messages are printed above the progress line.
/// </summary>
public sealed class PullProgressPrinter : IProgress<DownloadProgress>
{
    private static readonly TimeSpan s_terminalInterval = TimeSpan.FromMilliseconds(200);
    private static readonly TimeSpan s_redirectedInterval = TimeSpan.FromSeconds(5);
    private const int FileNameWidth = 40;

    private readonly TextWriter _output = Console.Error;
    private readonly bool _interactive = !Console.IsErrorRedirected;
    private readonly Stopwatch _clock = Stopwatch.StartNew();
    private readonly Lock _lock = new();
    private TimeSpan? _lastPrint;
    private int _lineLength;

    public void Report(DownloadProgress value)
    {
        lock (_lock)
        {
            var now = _clock.Elapsed;
            bool done = value.Received >= value.Total;
            if (!done && _lastPrint is { } last && now - last < (_interactive ? s_terminalInterval : s_redirectedInterval))
            {
                return;
            }

            _lastPrint = now;
            var line = Format(value, now);
            if (_interactive)
            {
                _output.Write('\r' + line.PadRight(_lineLength));
                _lineLength = line.Length;
                if (done)
                {
                    EndLine();
                }
            }
            else
            {
                _output.WriteLine(line);
            }
        }
    }

    /// <summary>Prints a log message on its own line.</summary>
    public void Log(string message)
    {
        lock (_lock)
        {
            EndLine();
            _output.WriteLine(message);
        }
    }

    private void EndLine()
    {
        if (_interactive && _lineLength > 0)
        {
            _output.WriteLine();
            _lineLength = 0;
        }
    }

    private static string Format(DownloadProgress value, TimeSpan elapsed)
    {
        double percent = value.Total > 0 ? 100.0 * value.Received / value.Total : 100;
        double speed = elapsed.TotalSeconds > 0 ? (value.Received - value.Resumed) / elapsed.TotalSeconds : 0;
        var eta = speed > 0 ? TimeSpan.FromSeconds((value.Total - value.Received) / speed) : (TimeSpan?)null;
        var etaText = eta is { } remaining ? $"ETA {remaining:hh\\:mm\\:ss}" : "ETA --:--:--";
        var file = value.File.Length > FileNameWidth ? "..." + value.File[^(FileNameWidth - 3)..] : value.File;
        return $"{file,-FileNameWidth} {ByteSize.Format(value.Received),10} / {ByteSize.Format(value.Total),-10} {percent,5:F1}%  "
            + $"{ByteSize.Format((long)speed),10}/s  {etaText}";
    }
}
