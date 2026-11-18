namespace Sharplight.Studio.ViewModels.Server;

/// <summary>An endpoint shown on the server page.</summary>
/// <param name="Method">HTTP method.</param>
/// <param name="Path">Path.</param>
/// <param name="Description">What it does.</param>
/// <param name="Models">The resident models reachable on it (empty for the endpoints that are not about one model).</param>
internal sealed record EndpointInfo(string Method, string Path, string Description, IReadOnlyList<string>? Models = null)
{
    /// <summary>The models as a line, or a note when none is resident; empty for endpoints that are not about models.</summary>
    public string ModelsText => Models is null ? "" : Models.Count == 0 ? "no resident model" : string.Join(", ", Models);
}
