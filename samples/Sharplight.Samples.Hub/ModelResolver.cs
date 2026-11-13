using Sharplight.Hub;

namespace Sharplight.Samples.Hub;

/// <summary>
/// Turns a model argument into a file: an existing file or directory is used as is, a stored model (<c>owner/name[:tag]</c>
/// or <c>alias[:tag]</c>) resolves to its package, and a Hub repository that is not stored yet is pulled first.
/// </summary>
public sealed class ModelResolver(HubContext hub, PullSettings pull)
{
    /// <summary>True when <paramref name="text"/> names an existing file or directory.</summary>
    public static bool IsLocalPath(string text) => File.Exists(text) || Directory.Exists(text);

    /// <exception cref="SampleException">The text names nothing that can be found or pulled, or pulling is disabled.</exception>
    public async Task<string> ResolveAsync(string text, CancellationToken cancellationToken)
    {
        if (IsLocalPath(text))
        {
            return text;
        }

        if (hub.Store.Find(text) is { } stored)
        {
            return stored.PackagePath;
        }

        if (!ModelReference.TryParse(text, out var reference))
        {
            throw new SampleException($"'{text}' is neither a file, a stored model (see 'list') nor a Hugging Face repository (owner/name[:tag]).",
                ExitCodes.InvalidInput);
        }

        if (!pull.AllowPull)
        {
            throw new SampleException($"'{text}' is not in the model store {hub.Store.Root}; pull it first or drop --no-pull.", ExitCodes.InvalidInput);
        }

        var printer = new PullProgressPrinter();
        printer.Log($"{reference} is not in the model store; pulling it.");
        var result = await hub.Puller.PullAsync(reference, pull.ToOptions(printer), cancellationToken);
        return result.Manifest.PackagePath;
    }
}
