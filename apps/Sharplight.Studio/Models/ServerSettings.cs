namespace Sharplight.Studio.Models;

/// <summary>Settings of the local OpenAI / Anthropic compatible endpoints.</summary>
internal sealed record ServerSettings
{
    /// <summary>Address to listen on (<c>127.0.0.1</c> keeps the server local to this machine).</summary>
    public string Host { get; init; } = "127.0.0.1";

    /// <summary>TCP port.</summary>
    public int Port { get; init; } = 8080;

    /// <summary>Required API key (<c>Authorization: Bearer</c> or <c>x-api-key</c>); empty means no authentication.</summary>
    public string ApiKey { get; init; } = "";

    /// <summary>Allowed CORS origins, comma separated (<c>*</c> for any); empty sends no CORS headers.</summary>
    public string CorsOrigins { get; init; } = "";

    /// <summary>Expose <c>/v1/chat/completions</c>, <c>/v1/completions</c> and <c>/v1/models</c>.</summary>
    public bool OpenAi { get; init; } = true;

    /// <summary>Expose <c>/v1/messages</c> and <c>/v1/messages/count_tokens</c>.</summary>
    public bool Anthropic { get; init; } = true;

    /// <summary>Ids of the loaded models the server exposes; empty serves every loaded model.</summary>
    public IReadOnlyList<string> ServedModels { get; init; } = [];
}
