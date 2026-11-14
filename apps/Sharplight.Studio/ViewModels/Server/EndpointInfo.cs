namespace Sharplight.Studio.ViewModels.Server;

/// <summary>An endpoint shown on the server page.</summary>
/// <param name="Method">HTTP method.</param>
/// <param name="Path">Path.</param>
/// <param name="Description">What it does.</param>
internal sealed record EndpointInfo(string Method, string Path, string Description);
