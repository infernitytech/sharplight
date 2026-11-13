using System.Text.Json;
using System.Text.Json.Nodes;

namespace Sharplight.AspNetCore;

/// <summary>
/// Writes the streamed chunk objects of one OpenAI response (<c>chat.completion.chunk</c> or <c>text_completion</c>) directly
/// as JSON into an <see cref="SseWriter"/>, without building a node tree per token.
/// </summary>
internal sealed class OpenAiChunkWriter(SseWriter sse, string id, string objectType, long created, string model, bool includeUsage)
{
    private static readonly JsonEncodedText IdName = JsonEncodedText.Encode("id");
    private static readonly JsonEncodedText ObjectName = JsonEncodedText.Encode("object");
    private static readonly JsonEncodedText CreatedName = JsonEncodedText.Encode("created");
    private static readonly JsonEncodedText ModelName = JsonEncodedText.Encode("model");
    private static readonly JsonEncodedText ChoicesName = JsonEncodedText.Encode("choices");
    private static readonly JsonEncodedText TextName = JsonEncodedText.Encode("text");
    private static readonly JsonEncodedText IndexName = JsonEncodedText.Encode("index");
    private static readonly JsonEncodedText DeltaName = JsonEncodedText.Encode("delta");
    private static readonly JsonEncodedText LogprobsName = JsonEncodedText.Encode("logprobs");
    private static readonly JsonEncodedText FinishName = JsonEncodedText.Encode("finish_reason");
    private static readonly JsonEncodedText UsageName = JsonEncodedText.Encode("usage");

    /// <summary>A legacy-completions chunk: <c>{"text": ..., "finish_reason": ...}</c>.</summary>
    public void Text(int index, string text, string? finish = null, JsonNode? logprobs = null)
    {
        var w = Begin();
        w.WriteString(TextName, text);
        End(w, index, finish, logprobs);
    }

    /// <summary>
    /// A chat chunk whose delta is <paramref name="delta"/>, or <c>{field: value}</c> when <paramref name="delta"/> is null and
    /// <paramref name="field"/> is set, or empty.
    /// </summary>
    public void Delta(int index, string? field, string? value, JsonObject? delta = null, string? finish = null, JsonNode? logprobs = null)
    {
        var w = Begin();
        w.WritePropertyName(DeltaName);
        if (delta is not null)
        {
            delta.WriteTo(w, HttpJson.Options);
        }
        else
        {
            w.WriteStartObject();
            if (field is not null)
            {
                w.WriteString(field, value);
            }

            w.WriteEndObject();
        }

        End(w, index, finish, logprobs);
    }

    /// <summary>The final usage-only chunk (<c>stream_options.include_usage</c>).</summary>
    public void Usage(JsonObject usage)
    {
        var w = sse.BeginEvent();
        w.WriteStartObject();
        WriteHeader(w);
        w.WriteStartArray(ChoicesName);
        w.WriteEndArray();
        w.WritePropertyName(UsageName);
        usage.WriteTo(w, HttpJson.Options);
        w.WriteEndObject();
        sse.EndEvent();
    }

    private Utf8JsonWriter Begin()
    {
        var w = sse.BeginEvent();
        w.WriteStartObject();
        WriteHeader(w);
        w.WriteStartArray(ChoicesName);
        w.WriteStartObject();
        return w;
    }

    private void WriteHeader(Utf8JsonWriter w)
    {
        w.WriteString(IdName, id);
        w.WriteString(ObjectName, objectType);
        w.WriteNumber(CreatedName, created);
        w.WriteString(ModelName, model);
    }

    private void End(Utf8JsonWriter w, int index, string? finish, JsonNode? logprobs)
    {
        w.WriteNumber(IndexName, index);
        w.WritePropertyName(LogprobsName);
        if (logprobs is null)
        {
            w.WriteNullValue();
        }
        else
        {
            logprobs.WriteTo(w, HttpJson.Options);
        }

        if (finish is null)
        {
            w.WriteNull(FinishName);
        }
        else
        {
            w.WriteString(FinishName, finish);
        }

        w.WriteEndObject();
        w.WriteEndArray();
        if (includeUsage)
        {
            w.WriteNull(UsageName);
        }

        w.WriteEndObject();
        sse.EndEvent();
    }
}
