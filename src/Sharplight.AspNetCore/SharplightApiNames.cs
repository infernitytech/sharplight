namespace Sharplight.AspNetCore;

/// <summary>
/// Short, stable names of the API operations for usage records and metrics (<see cref="SharplightUsage.Api"/>), independent of the route
/// group the endpoints are mapped into.
/// </summary>
public static class SharplightApiNames
{
    /// <summary>OpenAI <c>POST /v1/chat/completions</c>.</summary>
    public const string OpenAiChatCompletions = "openai.chat.completions";

    /// <summary>OpenAI <c>POST /v1/completions</c>.</summary>
    public const string OpenAiCompletions = "openai.completions";

    /// <summary>OpenAI <c>POST /v1/embeddings</c>.</summary>
    public const string OpenAiEmbeddings = "openai.embeddings";

    /// <summary>OpenAI <c>POST /v1/audio/transcriptions</c>.</summary>
    public const string OpenAiAudioTranscriptions = "openai.audio.transcriptions";

    /// <summary>OpenAI <c>POST /v1/audio/translations</c>.</summary>
    public const string OpenAiAudioTranslations = "openai.audio.translations";

    /// <summary>OpenAI <c>POST /v1/images/generations</c>.</summary>
    public const string OpenAiImagesGenerations = "openai.images.generations";

    /// <summary>Sharplight <c>POST /v1/timeseries/forecast</c>.</summary>
    public const string TimeSeriesForecast = "timeseries.forecast";

    /// <summary>Anthropic <c>POST /v1/messages</c>.</summary>
    public const string AnthropicMessages = "anthropic.messages";

    /// <summary>Anthropic <c>POST /v1/messages/count_tokens</c>.</summary>
    public const string AnthropicCountTokens = "anthropic.messages.count_tokens";

    /// <summary>Cohere / Jina style <c>POST /v1/rerank</c> (also <c>/v2/rerank</c>, <c>/rerank</c>).</summary>
    public const string Rerank = "rerank";

    /// <summary><c>GET /v1/models</c>.</summary>
    public const string ModelsList = "models.list";

    /// <summary><c>GET /v1/models/{id}</c>.</summary>
    public const string ModelsGet = "models.get";

    private static readonly (string Suffix, string Name)[] s_routes =
    [
        ("/v1/messages/count_tokens", AnthropicCountTokens),
        ("/v1/messages", AnthropicMessages),
        ("/v1/chat/completions", OpenAiChatCompletions),
        ("/v1/completions", OpenAiCompletions),
        ("/v1/embeddings", OpenAiEmbeddings),
        ("/v1/audio/transcriptions", OpenAiAudioTranscriptions),
        ("/v1/audio/translations", OpenAiAudioTranslations),
        ("/v1/images/generations", OpenAiImagesGenerations),
        ("/v1/timeseries/forecast", TimeSeriesForecast),
        ("/rerank", Rerank),
        ("/v1/models/{id}", ModelsGet),
        ("/v1/models", ModelsList),
    ];

    /// <summary>The name of the operation a route pattern serves (matched by its end, so any group prefix works); the pattern itself when unknown.</summary>
    /// <param name="routePattern">The endpoint's route pattern.</param>
    /// <returns>The name.</returns>
    public static string Of(string routePattern)
    {
        ArgumentNullException.ThrowIfNull(routePattern);
        foreach (var (suffix, name) in s_routes)
        {
            if (routePattern.EndsWith(suffix, StringComparison.Ordinal))
            {
                return name;
            }
        }

        return routePattern;
    }
}
