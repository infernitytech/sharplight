using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.AI;
using Sharplight.Llm.Serving.Chat;
using Sharplight.Llm.Structured;
using MeaiFinishReason = Microsoft.Extensions.AI.ChatFinishReason;
using ServingFinishReason = Sharplight.Llm.Serving.Chat.ChatFinishReason;

namespace Sharplight.Extensions.AI;

/// <summary>
/// <see cref="IChatClient"/> over the batched serving engine (<see cref="ModelService"/>, e.g. a chat model of a <c>ModelHost</c>): calls
/// are continuously batched with every other request of the model, queued fairly per <see cref="ServingChatClientOptions.Tenant"/> and
/// isolated in the prefix cache per <see cref="ServingChatClientOptions.CachePartition"/>. The model's chat template, reasoning and
/// tool-call parsing apply. Text, reasoning, tool calls and tool results are mapped, and images (<see cref="DataContent"/> with an image
/// media type) where the service takes image input (<see cref="ModelService.SupportsImages"/>; they are batched like text). Thread-safe; the
/// caller keeps ownership of the service (and of any lease that keeps it loaded).
/// </summary>
/// <example>
/// <code>
/// using var lease = await host.AcquireAsync("qwen3", ModelKind.Chat, ct);
/// var service = ((IChatModel)lease.As&lt;LoadedModel&gt;()).Service;
/// IChatClient client = new ServingChatClient(service, new ServingChatClientOptions { Tenant = "space-1", CachePartition = "space-1" });
/// var answer = await client.GetResponseAsync("Say hi", cancellationToken: ct);
/// </code>
/// </example>
public sealed class ServingChatClient : IChatClient
{
    /// <summary><see cref="ChatOptions.AdditionalProperties"/> key overriding the tenant of one call (a string).</summary>
    public const string TenantProperty = "sharplight.tenant";

    /// <summary><see cref="ChatOptions.AdditionalProperties"/> key overriding the prefix-cache partition of one call (a string).</summary>
    public const string CachePartitionProperty = "sharplight.cache_partition";

    /// <summary><see cref="ChatOptions.AdditionalProperties"/> key overriding the prefix-cache salt of one call (a string, see <see cref="GenerationRequest.CacheSalt"/>).</summary>
    public const string CacheSaltProperty = "sharplight.cache_salt";

    /// <summary><see cref="ChatOptions.AdditionalProperties"/> key overriding the priority of one call (an int).</summary>
    public const string PriorityProperty = "sharplight.priority";

    private readonly ModelService _service;
    private readonly ServingChatClientOptions _options;
    private readonly ChatClientMetadata _metadata;

    /// <summary>Creates the client.</summary>
    /// <param name="service">The serving chat service; the caller keeps ownership.</param>
    /// <param name="options">Tenant, partition and defaults, or null.</param>
    public ServingChatClient(ModelService service, ServingChatClientOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(service);
        _service = service;
        _options = options ?? new();
        _metadata = new ChatClientMetadata("sharplight", null, _options.ModelId ?? service.ModelName);
    }

    /// <summary>The service the client runs on.</summary>
    public ModelService Service => _service;

    /// <inheritdoc/>
    public async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        var updates = new List<ChatResponseUpdate>();
        await foreach (var update in GetStreamingResponseAsync(messages, options, cancellationToken).ConfigureAwait(false))
        {
            updates.Add(update);
        }

        return updates.ToChatResponse();
    }

    /// <inheritdoc/>
    /// <exception cref="InvalidOperationException">Generation failed or timed out.</exception>
    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var request = CreateRequest(messages, options);
        string id = "chatcmpl-" + Guid.NewGuid().ToString("N")[..16];
        var created = DateTimeOffset.UtcNow;
        string model = options?.ModelId ?? _metadata.DefaultModelId!;
        ChatResponseUpdate Update(IList<AIContent> contents) => new(ChatRole.Assistant, contents)
        {
            ResponseId = id,
            MessageId = id,
            CreatedAt = created,
            ModelId = model,
        };

        var run = _service.StartChat(request, cancellationToken);
        await foreach (var chat in run.ReadAllAsync(cancellationToken).ConfigureAwait(false))
        {
            if (chat.Content is { } text)
            {
                yield return Update([new TextContent(text)]);
            }
            else if (chat.Reasoning is { } reasoning)
            {
                yield return Update([new TextReasoningContent(reasoning)]);
            }
            else if (chat.ToolCall is { } call)
            {
                yield return Update([ChatOutputContents.From(call)]);
            }
            else if (chat.Finish is { } finish)
            {
                if (finish is ServingFinishReason.Error or ServingFinishReason.Timeout)
                {
                    throw new InvalidOperationException(chat.Error ?? $"Generation ended with {finish}.");
                }

                var usage = chat.Usage ?? default;
                var final = Update([new UsageContent(new UsageDetails
                {
                    InputTokenCount = usage.PromptTokens,
                    OutputTokenCount = usage.CompletionTokens,
                    TotalTokenCount = usage.TotalTokens,
                    CachedInputTokenCount = usage.CachedTokens,
                })]);
                final.FinishReason = finish switch
                {
                    ServingFinishReason.Length => MeaiFinishReason.Length,
                    ServingFinishReason.ToolCalls => MeaiFinishReason.ToolCalls,
                    _ => MeaiFinishReason.Stop,
                };
                yield return final;
            }
        }
    }

    /// <summary>Returns the metadata or the client itself (or the service) when assignable; null for keyed requests.</summary>
    /// <param name="serviceType">The requested type.</param>
    /// <param name="serviceKey">A key; keyed requests return null.</param>
    /// <returns>The service, or null.</returns>
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null)
        {
            return null;
        }

        if (serviceType == typeof(ChatClientMetadata))
        {
            return _metadata;
        }

        if (serviceType.IsInstanceOfType(this))
        {
            return this;
        }

        return serviceType.IsInstanceOfType(_service) ? _service : null;
    }

    /// <summary>Nothing to release: the caller owns the service.</summary>
    public void Dispose()
    {
        // The service and its engine belong to the caller.
    }

    /// <summary>The serving request of one call (internal for tests).</summary>
    internal ChatRequest CreateRequest(IEnumerable<ChatMessage> messages, ChatOptions? options)
    {
        var turns = messages.Select(ToTurn).ToList();
        if (_options.DefaultSystemPrompt is { } system && !turns.Any(turn => turn.Role == "system"))
        {
            turns.Insert(0, new ChatTurn("system", system));
        }

        var sampling = _options.DefaultSampling with
        {
            Temperature = options?.Temperature ?? _options.DefaultSampling.Temperature,
            TopP = options?.TopP ?? _options.DefaultSampling.TopP,
            TopK = options?.TopK ?? _options.DefaultSampling.TopK,
            Seed = options?.Seed is { } seed ? (int)seed : _options.DefaultSampling.Seed,
            FrequencyPenalty = options?.FrequencyPenalty ?? _options.DefaultSampling.FrequencyPenalty,
            PresencePenalty = options?.PresencePenalty ?? _options.DefaultSampling.PresencePenalty,
        };
        var tools = (options?.Tools ?? []).OfType<AIFunctionDeclaration>()
            .Select(tool => new ToolSpec(tool.Name, tool.Description, tool.JsonSchema.GetRawText())).ToList();
        var (mode, function) = options?.ToolMode switch
        {
            NoneChatToolMode => (ToolChoiceMode.None, (string?)null),
            RequiredChatToolMode { RequiredFunctionName: { } name } => (ToolChoiceMode.Function, name),
            RequiredChatToolMode => (ToolChoiceMode.Required, null),
            _ => (ToolChoiceMode.Auto, null),
        };
        return new ChatRequest
        {
            Messages = turns,
            Tools = tools,
            ToolChoice = mode,
            ToolChoiceFunction = function,
            Sampling = sampling,
            MaxTokens = options?.MaxOutputTokens ?? _options.DefaultMaxTokens,
            StopStrings = [.. options?.StopSequences ?? []],
            ResponseFormat = options?.ResponseFormat switch
            {
                ChatResponseFormatJson { Schema: { } schema } => OutputFormat.JsonSchema(schema.GetRawText()),
                ChatResponseFormatJson => OutputFormat.JsonObject,
                _ => null,
            },
            Tenant = Property<string>(options, TenantProperty) ?? _options.Tenant,
            CachePartition = Property<string>(options, CachePartitionProperty) ?? _options.CachePartition,
            CacheSalt = Property<string>(options, CacheSaltProperty) ?? _options.CacheSalt,
            Priority = options?.AdditionalProperties?.TryGetValue(PriorityProperty, out var priority) == true && priority is int value ? value : _options.Priority,
            Timeout = _options.Timeout,
        };
    }

    private static T? Property<T>(ChatOptions? options, string key)
        where T : class
        => options?.AdditionalProperties?.TryGetValue(key, out var value) == true ? value as T : null;

    private static ChatTurn ToTurn(ChatMessage message)
    {
        string text = string.Concat(message.Contents.OfType<TextContent>().Select(content => content.Text));
        if (message.Role == ChatRole.Tool)
        {
            var result = message.Contents.OfType<FunctionResultContent>().FirstOrDefault();
            return new ChatTurn("tool", result is null ? text : ResultText(result.Result), ToolCallId: result?.CallId);
        }

        if (message.Role == ChatRole.Assistant)
        {
            var calls = message.Contents.OfType<FunctionCallContent>()
                .Select(call => new ToolCall(call.CallId, call.Name, JsonSerializer.Serialize(call.Arguments ?? new Dictionary<string, object?>(), AIJsonUtilities.DefaultOptions.GetTypeInfo(typeof(IDictionary<string, object?>)))))
                .ToList();
            string reasoning = string.Concat(message.Contents.OfType<TextReasoningContent>().Select(content => content.Text));
            return new ChatTurn("assistant", text, Reasoning: reasoning.Length > 0 ? reasoning : null, ToolCalls: calls.Count > 0 ? calls : null);
        }

        var images = message.Contents.OfType<DataContent>().Where(content => content.HasTopLevelMediaType("image")).ToList();
        string role = message.Role == ChatRole.System ? "system" : "user";
        if (images.Count == 0)
        {
            return new ChatTurn(role, text);
        }

        // Images first, then the text, in the order the message lists them.
        var parts = message.Contents
            .Select(content => content switch
            {
                DataContent data when data.HasTopLevelMediaType("image") => ChatPart.FromImage(new ImageData(data.Data.ToArray(), data.MediaType, null)),
                TextContent t => ChatPart.FromText(t.Text),
                _ => null,
            })
            .OfType<ChatPart>()
            .ToList();
        return new ChatTurn(role, Parts: parts);
    }

    private static string ResultText(object? result) => result switch
    {
        null => "",
        string text => text,
        JsonElement element => element.ValueKind == JsonValueKind.String ? element.GetString()! : element.GetRawText(),
        _ => result.ToString() ?? "",
    };
}
