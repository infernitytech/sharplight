using System.CommandLine;

namespace Sharplight.Samples;

/// <summary>Runs a <c>System.CommandLine</c> command and maps failures to exit codes and one-line error messages.</summary>
public static class CommandRunner
{
    /// <summary>
    /// Parses <paramref name="args"/> and invokes <paramref name="command"/>. Ctrl+C cancels the token given to the command's
    /// action. Parse errors print usage and yield <see cref="ExitCodes.UsageError"/>.
    /// </summary>
    public static Task<int> RunAsync(Command command, string[] args)
    {
        return command.Parse(args).InvokeAsync();
    }

    /// <summary>
    /// Wraps a command action so that a <see cref="SampleException"/> is reported as <c>error: message</c> with
    /// <see cref="ExitCodes.RuntimeError"/> and cancellation (Ctrl+C) yields <see cref="ExitCodes.Cancelled"/>. Other exceptions
    /// are unexpected and propagate with a stack trace.
    /// </summary>
    public static Func<ParseResult, CancellationToken, Task<int>> Guard(Func<ParseResult, CancellationToken, Task<int>> action)
    {
        return async (parseResult, cancellationToken) =>
        {
            try
            {
                return await action(parseResult, cancellationToken);
            }
            catch (SampleException error)
            {
                await Console.Error.WriteLineAsync($"error: {error.Message}");
                return ExitCodes.RuntimeError;
            }
            catch (OperationCanceledException)
            {
                await Console.Error.WriteLineAsync("cancelled");
                return ExitCodes.Cancelled;
            }
        };
    }
}
