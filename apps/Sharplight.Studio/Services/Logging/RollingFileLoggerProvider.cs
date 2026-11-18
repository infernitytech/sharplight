using System.Globalization;
using System.Text;
using Microsoft.Extensions.Logging;

namespace Sharplight.Studio.Services.Logging;

/// <summary>
/// Writes log lines to <see cref="FileLogOptions.FileName"/> in the log folder and rolls the file over at the size limit, keeping the
/// newest <see cref="FileLogOptions.MaxFiles"/> files. Logging never throws: a folder that cannot be written silently disables the file.
/// Callers log events and exceptions only, never prompts or other user content.
/// </summary>
internal sealed class RollingFileLoggerProvider : ILoggerProvider
{
    private readonly FileLogOptions _options;
    private readonly object _gate = new();
    private long _size;
    private bool _opened;
    private bool _disabled;

    /// <summary>Creates the provider; the folder is created with the first line.</summary>
    public RollingFileLoggerProvider(FileLogOptions options)
    {
        _options = options;
    }

    /// <summary>The path of the current log file.</summary>
    public string CurrentFile => Path.Combine(_options.Directory, _options.FileName);

    /// <inheritdoc/>
    public ILogger CreateLogger(string categoryName) => new FileLogger(this, categoryName, _options.MinimumLevel);

    /// <inheritdoc/>
    public void Dispose()
    {
    }

    /// <summary>Appends one formatted entry.</summary>
    internal void Write(LogLevel level, string category, string message, Exception? error)
    {
        var text = new StringBuilder();
        text.Append(DateTimeOffset.Now.ToString("yyyy-MM-dd HH:mm:ss.fff zzz", CultureInfo.InvariantCulture));
        text.Append(' ').Append(Abbreviate(level)).Append(' ').Append(category).Append(": ").Append(message);
        if (error is not null)
        {
            text.AppendLine().Append(error);
        }

        text.AppendLine();
        var bytes = Encoding.UTF8.GetBytes(text.ToString());
        lock (_gate)
        {
            if (_disabled)
            {
                return;
            }

            try
            {
                if (!_opened)
                {
                    Directory.CreateDirectory(_options.Directory);
                    _size = File.Exists(CurrentFile) ? new FileInfo(CurrentFile).Length : 0;
                    _opened = true;
                }

                if (_size > 0 && _size + bytes.Length > _options.MaxFileBytes)
                {
                    Roll();
                }

                using (var stream = new FileStream(CurrentFile, FileMode.Append, FileAccess.Write, FileShare.ReadWrite))
                {
                    stream.Write(bytes);
                }

                _size += bytes.Length;
            }
            catch (Exception error2) when (error2 is IOException or UnauthorizedAccessException)
            {
                _disabled = true;
            }
        }
    }

    private void Roll()
    {
        var stem = Path.GetFileNameWithoutExtension(_options.FileName);
        var extension = Path.GetExtension(_options.FileName);
        string Rolled(int index) => Path.Combine(_options.Directory, $"{stem}.{index}{extension}");

        var keep = Math.Max(1, _options.MaxFiles);
        // The oldest rolled file goes, the others move up by one; keep = 1 keeps only the current file (truncated).
        for (var index = keep; index < keep + 8; index++)
        {
            if (File.Exists(Rolled(index)))
            {
                File.Delete(Rolled(index));
            }
        }

        if (keep == 1)
        {
            File.Delete(CurrentFile);
        }
        else
        {
            for (var index = keep - 2; index >= 1; index--)
            {
                if (File.Exists(Rolled(index)))
                {
                    File.Move(Rolled(index), Rolled(index + 1), overwrite: true);
                }
            }

            File.Move(CurrentFile, Rolled(1), overwrite: true);
        }

        _size = 0;
    }

    private static string Abbreviate(LogLevel level) => level switch
    {
        LogLevel.Trace => "TRC",
        LogLevel.Debug => "DBG",
        LogLevel.Information => "INF",
        LogLevel.Warning => "WRN",
        LogLevel.Error => "ERR",
        LogLevel.Critical => "CRT",
        _ => "---",
    };
}
