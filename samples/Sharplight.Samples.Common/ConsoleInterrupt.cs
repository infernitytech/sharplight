namespace Sharplight.Samples;

/// <summary>
/// A stretch of work that Ctrl+C interrupts on its own, without cancelling the whole command (e.g. one answer of an interactive
/// chat). While a scope is active, Ctrl+C cancels <see cref="Token"/> only; outside any scope it cancels the command
/// (see <see cref="CommandRunner.RunAsync"/>). Scopes do not nest: the latest one receives the interrupt.
/// </summary>
public sealed class ConsoleInterrupt : IDisposable
{
    private static ConsoleInterrupt? s_current;

    private readonly CancellationTokenSource _source;

    private ConsoleInterrupt(CancellationToken commandToken)
    {
        _source = CancellationTokenSource.CreateLinkedTokenSource(commandToken);
    }

    /// <summary>Cancelled by Ctrl+C during the scope, or when the command is cancelled.</summary>
    public CancellationToken Token => _source.Token;

    /// <summary>True when Ctrl+C (not the command's cancellation) ended the scope's work.</summary>
    public bool Interrupted { get; private set; }

    /// <summary>Starts a scope linked to the command's token.</summary>
    /// <param name="commandToken">The token of the running command.</param>
    public static ConsoleInterrupt Begin(CancellationToken commandToken)
    {
        var scope = new ConsoleInterrupt(commandToken);
        Volatile.Write(ref s_current, scope);
        return scope;
    }

    /// <summary>Ends the scope; Ctrl+C cancels the command again.</summary>
    public void Dispose()
    {
        Interlocked.CompareExchange(ref s_current, null, this);
        _source.Dispose();
    }

    /// <summary>Interrupts the active scope, if any; false when Ctrl+C should cancel the command.</summary>
    internal static bool TryInterrupt()
    {
        var scope = Volatile.Read(ref s_current);
        if (scope is null || scope.Interrupted)
        {
            return false;
        }

        scope.Interrupted = true;
        try
        {
            scope._source.Cancel();
        }
        catch (ObjectDisposedException)
        {
            return false;
        }

        return true;
    }
}
