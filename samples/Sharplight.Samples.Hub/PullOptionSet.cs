using System.CommandLine;
using Sharplight.Hub;

namespace Sharplight.Samples.Hub;

/// <summary>
/// The pull options: <c>--revision</c>, <c>--keep-source</c>, <c>--no-precompile</c>, plus <c>--force</c> for <c>pull</c> and
/// <c>--no-pull</c> for the commands that pull missing models automatically (<c>run</c>, <c>serve</c>).
/// </summary>
public sealed class PullOptionSet(bool explicitPull)
{
    private const string DefaultRevision = "main";

    private readonly Option<string> _revision = new("--revision")
    {
        HelpName = "rev",
        Description = "Branch, tag or commit to pull; it is resolved to a commit that every file is pinned to.",
        DefaultValueFactory = _ => DefaultRevision,
    };

    private readonly Option<bool> _keepSource = new("--keep-source")
    {
        Description = "Keep the downloaded GGUF/safetensors/ONNX files next to the package instead of deleting them.",
    };

    private readonly Option<bool> _noPrecompile = new("--no-precompile")
    {
        Description = "Do not precompile the package for the local GPU (kernels are then compiled at the first load).",
    };

    private readonly Option<bool> _force = new("--force")
    {
        Description = "Download and convert again even when the stored model is at the latest commit.",
    };

    private readonly Option<bool> _noPull = new("--no-pull")
    {
        Description = "Fail instead of pulling models that are not in the model store.",
    };

    public void AddTo(Command command)
    {
        command.Options.Add(_revision);
        command.Options.Add(_keepSource);
        command.Options.Add(_noPrecompile);
        command.Options.Add(explicitPull ? _force : _noPull);
    }

    /// <param name="parseResult">The parsed command line.</param>
    /// <param name="precompile">Creates the precompile target when precompiling is enabled; receives the log callback.</param>
    public PullSettings Read(ParseResult parseResult, Func<Action<string>, PrecompileTarget?> precompile)
    {
        bool enabled = !parseResult.GetValue(_noPrecompile);
        return new PullSettings(
            explicitPull || !parseResult.GetValue(_noPull),
            parseResult.GetValue(_revision)!,
            parseResult.GetValue(_keepSource),
            explicitPull && parseResult.GetValue(_force),
            log => enabled ? precompile(log) : null);
    }
}
