using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;
using Sharplight.Llm;
using Sharplight.Templating;
using ChatMessage = Microsoft.Extensions.AI.ChatMessage;

namespace Sharplight.Extensions.AI;

/// <summary>
/// <see cref="IChatClient"/> over a Sharplight model. Messages are rendered with the model's chat template, tool
/// definitions are passed to the template (when it supports <c>tools</c>) and tool calls in the output are parsed into
/// <see cref="FunctionCallContent"/> (wrap with <c>UseFunctionInvocation()</c> to execute them). Images (as
/// <see cref="DataContent"/>) are supported when the model has a vision tower.
/// <para>
/// Thread safety: instances may be shared; whether calls run concurrently is up to the <see cref="ITextGenerationBackend"/>
/// (the default <see cref="SessionGenerationBackend"/> serializes them). Dispose the client to release the model when it owns it.
/// </para>
/// </summary>
/// <example>
/// <code>
/// GgufImporter.Register();
/// using var client = SharplightChatClient.Load("qwen3-4b-q4_k_m.gguf", new LlmLoadOptions { Devices = [device] });
/// var response = await client.GetResponseAsync("Why is the sky blue?");
/// Console.WriteLine(response.Text);
/// </code>
/// </example>
public sealed class SharplightChatClient : IChatClient
{
    private readonly ITextGenerationBackend _backend;
    private readonly SharplightChatClientOptions _options;
    private readonly bool _ownsBackend;
    private readonly ChatClientMetadata _metadata;
    private ChatTemplate? _template;

    /// <summary>Creates a client over an arbitrary backend (for example a shared continuous-batching engine).</summary>
    /// <param name="backend">Supplies the chat template and the text stream.</param>
    /// <param name="options">Client configuration; defaults when null.</param>
    /// <param name="ownsBackend">Dispose <paramref name="backend"/> together with the client.</param>
    public SharplightChatClient(ITextGenerationBackend backend, SharplightChatClientOptions? options = null, bool ownsBackend = false)
    {
        _backend = backend;
        _options = options ?? new();
        _ownsBackend = ownsBackend;
        _metadata = new ChatClientMetadata("sharplight", null, _options.ModelName ?? backend.ModelName);
    }

    /// <summary>
    /// Wraps a loaded model (one generation at a time): a <see cref="SessionGenerationBackend"/>, or a <see cref="DiffusionGenerationBackend"/>
    /// for a masked-diffusion model. The model is not disposed with the client unless <paramref name="ownsModel"/>.
    /// </summary>
    public SharplightChatClient(LlmModel model, SharplightChatClientOptions? options = null, bool ownsModel = false)
        : this(model.IsDiffusion ? new DiffusionGenerationBackend(model, ownsModel) : new SessionGenerationBackend(model, ownsModel), options, ownsBackend: true) { }

    /// <summary>Loads a model and returns a client owning it (importers must be registered, e.g. <c>GgufImporter.Register()</c>).</summary>
    public static SharplightChatClient Load(string path, LlmLoadOptions load, SharplightChatClientOptions? options = null)
        => new(LlmModel.Load(path, load), options, ownsModel: true);

    /// <summary>The backend that runs the generations.</summary>
    public ITextGenerationBackend Backend => _backend;

    /// <summary>Runs one chat turn to completion; equivalent to folding <see cref="GetStreamingResponseAsync"/>.</summary>
    /// <exception cref="NotSupportedException">The messages contain images or content the model cannot handle.</exception>
    /// <exception cref="InvalidOperationException">The model has no chat template, or the prompt exceeds the context.</exception>
    public async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        var updates = new List<ChatResponseUpdate>();
        await foreach (var u in GetStreamingResponseAsync(messages, options, cancellationToken).ConfigureAwait(false))
        {
            updates.Add(u);
        }

        return updates.ToChatResponse();
    }

    /// <summary>
    /// Streams one chat turn. Text and reasoning arrive incrementally, tool calls as complete <see cref="FunctionCallContent"/>,
    /// and the last update carries <see cref="UsageContent"/> and the finish reason.
    /// </summary>
    /// <inheritdoc cref="GetResponseAsync"/>
    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var template = _template ??= CreateTemplate();
        var req = PromptBuilder.Prepare(template, messages, options, _options, _backend.SupportsImages);
        var parser = new OutputParser(req.StartsInReasoning, req.Options.StopStrings,
            req.HasTools ? _options.ToolCallStart : null, _options.ToolCallEnd, _options.ToolCallParsers, req.Tools);

        string id = "chatcmpl-" + Guid.NewGuid().ToString("N")[..16];
        var created = DateTimeOffset.UtcNow;
        string model = options?.ModelId ?? _metadata.DefaultModelId ?? _backend.ModelName;
        ChatResponseUpdate Update(IList<AIContent>? contents = null) => new(ChatRole.Assistant, contents ?? [])
        {
            ResponseId = id,
            MessageId = id,
            CreatedAt = created,
            ModelId = model,
        };

        var result = new TextGenerationResult();
        var genRequest = new TextGenerationRequest { Prompt = req.Prompt, Images = req.Images, Options = req.Options };
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        await foreach (var piece in _backend.GenerateAsync(genRequest, result, cts.Token).ConfigureAwait(false))
        {
            var contents = parser.Push(piece);
            if (contents.Count > 0)
            {
                yield return Update([.. contents]);
            }

            if (parser.Stopped)
            {
                cts.Cancel();
                break;
            }
        }

        var tail = parser.Complete();
        if (tail.Count > 0)
        {
            yield return Update([.. tail]);
        }

        var final = Update([new UsageContent(new UsageDetails
        {
            InputTokenCount = result.PromptTokens,
            OutputTokenCount = result.GeneratedTokens,
            TotalTokenCount = result.PromptTokens + result.GeneratedTokens,
        })]);
        final.FinishReason = parser.ToolCallCount > 0 ? ChatFinishReason.ToolCalls
            : !parser.Stopped && result.ReachedTokenLimit ? ChatFinishReason.Length : ChatFinishReason.Stop;
        yield return final;
    }

    private ChatTemplate CreateTemplate()
    {
        var src = _backend.ChatTemplate ?? throw new InvalidOperationException("The model has no chat template; SharplightChatClient needs one to format messages.");
        return new ChatTemplate(src, _backend.BosToken, _backend.EosToken);
    }

    /// <summary>Returns the <see cref="ChatClientMetadata"/>, the client itself or the backend when assignable to <paramref name="serviceType"/>; null for keyed requests.</summary>
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

        if (serviceType.IsInstanceOfType(_backend))
        {
            return _backend;
        }

        return null;
    }

    /// <summary>Disposes the backend when the client owns it.</summary>
    public void Dispose()
    {
        if (_ownsBackend)
        {
            _backend.Dispose();
        }
    }
}
