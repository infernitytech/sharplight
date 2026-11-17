namespace Sharplight.Studio.Services.Updates;

/// <summary>The release stream a user follows.</summary>
internal enum UpdateChannelKind
{
    /// <summary>Tagged releases.</summary>
    Stable,

    /// <summary>Preview builds from the development branch (channel suffix <c>unstable</c>).</summary>
    Preview,
}
