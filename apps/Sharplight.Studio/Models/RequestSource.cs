namespace Sharplight.Studio.Models;

/// <summary>Where a generation request came from.</summary>
internal enum RequestSource
{
    /// <summary>The chat in the app.</summary>
    Chat,

    /// <summary>A client of the OpenAI / Anthropic endpoints.</summary>
    Api,
}
