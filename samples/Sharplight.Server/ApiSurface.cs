namespace Sharplight.Server;

/// <summary>Which HTTP API dialects the server exposes.</summary>
internal enum ApiSurface
{
    /// <summary>OpenAI-compatible endpoints under <c>/v1</c>.</summary>
    OpenAi,

    /// <summary>Anthropic-compatible <c>/v1/messages</c>.</summary>
    Anthropic,

    /// <summary>Both dialects.</summary>
    Both,
}
