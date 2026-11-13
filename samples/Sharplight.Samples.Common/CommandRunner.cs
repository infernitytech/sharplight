using System.CommandLine;

namespace Sharplight.Samples;

/// <summary>Runs a <c>System.CommandLine</c> command and maps failures to exit codes and one-line error messages.</summary>
public static class CommandRunner
{
    /// <summary>How long a cancelled command may take to finish before the process exits anyway.</summary>
    private static readonly TimeSpan s_terminationTimeout = TimeSpan.FromSeconds(2);

    /// <summary>
    /// Parses <paramref name="args"/> and invokes <paramref name="command"/>. Ctrl+C interrupts the active
    /// <see cref="ConsoleInterrupt"/> scope if there is one, else cancels the token given to the command's action (a command that
    /// does not finish within two seconds, or a second Ctrl+C, ends the process with <see cref="ExitCodes.Cancelled"/>). Parse
    /// errors print usage and yield <see cref="ExitCodes.UsageError"/>.
    /// </summary>
    public static async Task<int> RunAsync(Command command, string[] args)
    {
        using var cancellation = new CancellationTokenSource();
        void OnCancelKeyPress(object? sender, ConsoleCancelEventArgs e)
        {
            e.Cancel = true;
            if (ConsoleInterrupt.TryInterrupt())
            {
                return;
            }

            if (cancellation.IsCancellationRequested)
            {
                Environment.Exit(ExitCodes.Cancelled);
            }

            cancellation.Cancel();
            _ = Task.Delay(s_terminationTimeout).ContinueWith(_ => Environment.Exit(ExitCodes.Cancelled), TaskScheduler.Default);
        }

        Console.CancelKeyPress += OnCancelKeyPress;
        try
        {
            // System.CommandLine's own Ctrl+C handling is off: the handler above routes it.
            return await command.Parse(args).InvokeAsync(new InvocationConfiguration { ProcessTerminationTimeout = null }, cancellation.Token);
        }
        finally
        {
            Console.CancelKeyPress -= OnCancelKeyPress;
        }
    }

    /// <summary>
    /// Wraps a command action so that a <see cref="SampleException"/> is reported as <c>error: message</c> with
    /// <see cref="SampleException.ExitCode"/> and cancellation (Ctrl+C) yields <see cref="ExitCodes.Cancelled"/>. Other exceptions
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
                return error.ExitCode;
            }
            catch (OperationCanceledException)
            {
                await Console.Error.WriteLineAsync("cancelled");
                return ExitCodes.Cancelled;
            }
        };
    }
}
