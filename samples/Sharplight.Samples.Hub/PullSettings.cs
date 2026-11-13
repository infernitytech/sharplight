using Sharplight.Hub;

namespace Sharplight.Samples.Hub;

/// <summary>How models are pulled, by <c>pull</c> or automatically by <c>run</c> and <c>serve</c>.</summary>
/// <param name="AllowPull">Pull models that are not stored (false: fail instead).</param>
/// <param name="Revision">Branch, tag or commit to pull.</param>
/// <param name="KeepSource">Keep the downloaded source files next to the package.</param>
/// <param name="Force">Convert again even when the stored model is at the resolved commit.</param>
/// <param name="Precompile">Factory for the device to precompile for (null result: no precompile); receives the log callback.</param>
public sealed record PullSettings(bool AllowPull, string Revision, bool KeepSource, bool Force, Func<Action<string>, PrecompileTarget?> Precompile)
{
    /// <summary>Pull options that report to <paramref name="printer"/>.</summary>
    public PullOptions ToOptions(PullProgressPrinter printer)
    {
        return new PullOptions
        {
            Revision = Revision,
            KeepSource = KeepSource,
            Force = Force,
            Precompile = Precompile(printer.Log),
            Log = printer.Log,
            Progress = printer,
        };
    }
}
