using System.Text;
using System.Text.Json;
using System.Threading.Channels;
using Sharplight.Templating;
using Sharplight.Tokenizers;

namespace Sharplight.Llm.Serving.Chat;

public sealed record ModelServiceOptions
{
    /// <summary>Name reported by the APIs.</summary>
    public string ModelName { get; init; } = "sharplight";
    /// <summary>Completion length when a request gives none (0 = until the context is full).</summary>
    public int DefaultMaxTokens { get; init; } = 1024;
    /// <summary>Tool-call formats to parse; null = detected from the chat template.</summary>
    public IReadOnlyList<IToolCallFormat>? ToolFormats { get; init; }
    /// <summary>Overrides the model's own chat template (Jinja source).</summary>
    public string? ChatTemplate { get; init; }
}

/// <summary>A rendered and tokenized chat prompt.</summary>
public sealed record ChatPrompt(string Text, int[] Tokens, bool StartsInReasoning);

/// <summary>
/// The chat layer on top of an <see cref="InferenceEngine"/>: renders messages (with tools) through the model's Jinja chat
/// template, submits the requests and turns the generated text into content / reasoning / tool-call events.
/// </summary>
public sealed class ModelService
{
    readonly ChatTemplate? _template;
    readonly IReadOnlyList<IToolCallFormat> _formats;
    readonly string? _bos;

    public InferenceEngine Engine { get; }
    public ModelServiceOptions Options { get; }
    public ITokenizer Tokenizer => Engine.Tokenizer;
    public string ModelName => Options.ModelName;
    public bool HasChatTemplate => _template is not null;

    public ModelService(InferenceEngine engine, string? chatTemplate, ModelServiceOptions? options = null)
    {
        Engine = engine;
        Options = options ?? new();
        var src = Options.ChatTemplate ?? chatTemplate;
        string? Tok(int? id) => id is int i ? engine.Tokenizer.TokenString(i) : null;
        _bos = Tok(engine.Tokenizer.BosId);
        if (!string.IsNullOrWhiteSpace(src)) _template = new ChatTemplate(src, _bos, Tok(engine.Tokenizer.EosId));
        _formats = Options.ToolFormats ?? ToolCallFormats.ForTemplate(src);
    }

    // ---- prompt -----------------------------------------------------------------------------------------------------

    public ChatPrompt BuildPrompt(ChatRequest r)
    {
        foreach (var m in r.Messages)
            if (m.Parts?.Any(p => p.Type == "image") == true)
                throw new EngineException("Image inputs are not supported by this serving configuration.", "invalid_request");
        bool useTools = r.Tools.Count > 0 && r.ToolChoice != ToolChoiceMode.None;
        var tools = useTools ? r.Tools.Where(t => r.ToolChoice != ToolChoiceMode.Function || t.Name == r.ToolChoiceFunction).Select(ToDefinition).ToList() : null;
        if (useTools && tools!.Count == 0) tools = [.. r.Tools.Select(ToDefinition)];
        var msgs = r.Messages.Select(ToMessage).ToList();
        if (r.JsonMode)
        {
            const string hint = "Respond only with a single valid JSON object and no other text.";
            int sys = msgs.FindIndex(m => m.Role == "system");
            if (sys >= 0) msgs[sys] = msgs[sys] with { Content = (msgs[sys].Content ?? "") + "\n\n" + hint };
            else msgs.Insert(0, new Templating.ChatMessage("system", hint));
        }
        Dictionary<string, object?>? ctx = null;
        if (r.EnableThinking is bool et) (ctx ??= [])["enable_thinking"] = et;
        if (r.TemplateContext is not null) foreach (var kv in r.TemplateContext) (ctx ??= [])[kv.Key] = kv.Value;

        string text = _template is not null
            ? _template.Render(msgs, addGenerationPrompt: true, ctx, tools)
            : FallbackPrompt(msgs);
        bool addSpecial = Tokenizer.AddBos && _bos is not null && !text.StartsWith(_bos, StringComparison.Ordinal);
        var tokens = Tokenizer.Encode(text, addSpecial, parseSpecial: true);
        return new ChatPrompt(text, tokens, text.TrimEnd().EndsWith("<think>", StringComparison.Ordinal));
    }

    public int CountTokens(ChatRequest r) => BuildPrompt(r).Tokens.Length;

    static string FallbackPrompt(List<Templating.ChatMessage> msgs)
    {
        var sb = new StringBuilder();
        foreach (var m in msgs) sb.Append(char.ToUpperInvariant(m.Role[0])).Append(m.Role[1..]).Append(": ").Append(m.Content).Append('\n');
        return sb.Append("Assistant:").ToString();
    }

    static ToolDefinition ToDefinition(ToolSpec t)
    {
        IReadOnlyDictionary<string, object?>? pars = null;
        if (!string.IsNullOrWhiteSpace(t.ParametersJson) && JsonClr.Parse(t.ParametersJson) is JinjaDict d)
            pars = d.ToDictionary(kv => (string)kv.Key, kv => kv.Value);
        return new ToolDefinition(t.Name, t.Description, pars);
    }

    static Templating.ChatMessage ToMessage(ChatTurn t)
    {
        string? content = t.Content;
        IReadOnlyList<Templating.ChatContentPart>? parts = null;
        if (t.Parts is { Count: > 0 } ps)
        {
            if (ps.All(p => p.Type == "text")) content = string.Concat(ps.Select(p => p.Text));
            else parts = [.. ps.Select(p => p.Type == "text" ? Templating.ChatContentPart.FromText(p.Text ?? "") : Templating.ChatContentPart.Image())];
        }
        var calls = t.ToolCalls?.Select(c => new ChatToolCall(c.Id, c.Name, JsonClr.ParseObject(c.ArgumentsJson))).ToList();
        return new Templating.ChatMessage(t.Role, content ?? (parts is null && calls is null ? "" : null), parts, t.Reasoning, calls, t.Name, t.ToolCallId);
    }

    // ---- running ----------------------------------------------------------------------------------------------------

    /// <summary>Submits a chat request (N independent choices). Admission errors are thrown here, before any output exists.</summary>
    public ChatRun StartChat(ChatRequest r, CancellationToken ct = default)
    {
        var prompt = BuildPrompt(r);
        bool useTools = r.Tools.Count > 0 && r.ToolChoice != ToolChoiceMode.None;
        var names = r.Tools.Select(t => t.Name).ToList();
        return Start(prompt.Tokens, r.N, r.Sampling, r.MaxTokens, r.StopStrings, r.StopTokenIds, r.Priority, r.Timeout, r.Logprobs, r.TopLogprobs,
            () => new ChatOutputParser(useTools ? _formats : null, useTools ? names : null, prompt.StartsInReasoning), ct);
    }

    /// <summary>Submits a raw text completion request.</summary>
    public ChatRun StartCompletion(CompletionRequest r, CancellationToken ct = default)
    {
        var tokens = r.PromptTokens ?? Tokenizer.Encode(r.Prompt ?? "", addSpecial: true, parseSpecial: true);
        return Start(tokens, r.N, r.Sampling, r.MaxTokens, r.StopStrings, r.StopTokenIds, r.Priority, r.Timeout, r.Logprobs, r.TopLogprobs, null, ct);
    }

    ChatRun Start(int[] tokens, int n, SamplingParams sampling, int maxTokens, IReadOnlyList<string> stops, IReadOnlyList<int> stopIds, int priority,
        TimeSpan? timeout, bool logprobs, int topLogprobs, Func<ChatOutputParser>? parser, CancellationToken ct)
    {
        n = Math.Max(1, n);
        var handles = new List<GenerationHandle>(n);
        try
        {
            for (int i = 0; i < n; i++)
                handles.Add(Engine.Submit(new GenerationRequest
                {
                    PromptTokens = tokens,
                    Sampling = sampling.Seed is int s && i > 0 ? sampling with { Seed = s + i } : sampling,
                    MaxTokens = maxTokens > 0 ? maxTokens : Options.DefaultMaxTokens,
                    StopStrings = stops, StopTokenIds = stopIds, Priority = priority, Timeout = timeout,
                    Logprobs = logprobs, TopLogprobs = topLogprobs,
                }, ct));
        }
        catch { foreach (var h in handles) h.Cancel(); throw; }
        return new ChatRun(handles, parser, tokens.Length);
    }
}

/// <summary>A submitted chat/completion request: stream <see cref="ChatEvent"/>s or wait for the <see cref="ChatResult"/>.</summary>
public sealed class ChatRun(IReadOnlyList<GenerationHandle> handles, Func<ChatOutputParser>? parserFactory, int promptTokens)
{
    public int PromptTokens { get; } = promptTokens;
    public int ChoiceCount => handles.Count;

    public void Cancel() { foreach (var h in handles) h.Cancel(); }

    /// <summary>Streams events of all choices (interleaved when N &gt; 1). Abandoning the enumeration cancels the requests.</summary>
    public async IAsyncEnumerable<ChatEvent> ReadAllAsync([System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        if (handles.Count == 1)
        {
            await foreach (var e in Choice(0, ct).ConfigureAwait(false)) yield return e;
            yield break;
        }
        var ch = Channel.CreateUnbounded<ChatEvent>(new UnboundedChannelOptions { SingleReader = true });
        var tasks = Enumerable.Range(0, handles.Count).Select(i => Task.Run(async () =>
        {
            await foreach (var e in Choice(i, ct).ConfigureAwait(false)) ch.Writer.TryWrite(e);
        }, CancellationToken.None)).ToArray();
        _ = Task.WhenAll(tasks).ContinueWith(t => ch.Writer.TryComplete(t.Exception?.GetBaseException()), TaskScheduler.Default);
        try
        {
            await foreach (var e in ch.Reader.ReadAllAsync(ct).ConfigureAwait(false)) yield return e;
        }
        finally { Cancel(); }
    }

    async IAsyncEnumerable<ChatEvent> Choice(int index, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct)
    {
        var parser = parserFactory?.Invoke();
        await foreach (var c in handles[index].ReadAllAsync(ct).ConfigureAwait(false))
        {
            if (c.Text.Length > 0)
            {
                if (parser is null) yield return new ChatEvent(index, Content: c.Text);
                else foreach (var e in Map(index, parser.Push(c.Text))) yield return e;
            }
            if (c.Logprobs is not null) yield return new ChatEvent(index, Logprobs: c.Logprobs);
            if (c.Finish is { } f)
            {
                if (parser is not null) foreach (var e in Map(index, parser.Finish())) yield return e;
                var reason = f switch
                {
                    FinishReason.Stop => parser?.ToolCallCount > 0 ? ChatFinishReason.ToolCalls : ChatFinishReason.Stop,
                    FinishReason.Length => ChatFinishReason.Length,
                    FinishReason.Cancelled => ChatFinishReason.Cancelled,
                    FinishReason.Timeout => ChatFinishReason.Timeout,
                    _ => ChatFinishReason.Error,
                };
                yield return new ChatEvent(index, Finish: reason, Usage: c.Usage, StopSequence: c.StopSequence, Error: c.Error);
            }
        }
    }

    static IEnumerable<ChatEvent> Map(int index, List<ParsedPiece> pieces)
    {
        foreach (var p in pieces)
            yield return p.Kind switch
            {
                PieceKind.Content => new ChatEvent(index, Content: p.Text),
                PieceKind.Reasoning => new ChatEvent(index, Reasoning: p.Text),
                _ => new ChatEvent(index, ToolCall: p.Call),
            };
    }

    public async Task<ChatResult> CompleteAsync(CancellationToken ct = default)
    {
        int n = handles.Count;
        var content = Enumerable.Range(0, n).Select(_ => new StringBuilder()).ToArray();
        var reasoning = Enumerable.Range(0, n).Select(_ => new StringBuilder()).ToArray();
        var calls = Enumerable.Range(0, n).Select(_ => new List<ToolCall>()).ToArray();
        var lps = Enumerable.Range(0, n).Select(_ => new List<TokenLogprob>()).ToArray();
        var choices = new ChatChoice?[n];
        await foreach (var e in ReadAllAsync(ct).ConfigureAwait(false))
        {
            int i = e.Index;
            if (e.Content is not null) content[i].Append(e.Content);
            if (e.Reasoning is not null) reasoning[i].Append(e.Reasoning);
            if (e.ToolCall is not null) calls[i].Add(e.ToolCall);
            if (e.Logprobs is not null) lps[i].AddRange(e.Logprobs);
            if (e.Finish is { } f)
                choices[i] = new ChatChoice
                {
                    Index = i, Content = content[i].ToString(), Reasoning = reasoning[i].Length > 0 ? reasoning[i].ToString() : null,
                    ToolCalls = calls[i], Finish = f, StopSequence = e.StopSequence, Error = e.Error, Logprobs = lps[i], Usage = e.Usage ?? default,
                };
        }
        var done = choices.Select(c => c ?? throw new InvalidOperationException("A choice ended without a finish event.")).ToList();
        return new ChatResult
        {
            Choices = done,
            Usage = new GenerationUsage(done[0].Usage.PromptTokens, done.Sum(c => c.Usage.CompletionTokens), done[0].Usage.CachedTokens),
        };
    }
}

/// <summary>JSON to template-value conversion (objects become <see cref="JinjaDict"/>, numbers long/double).</summary>
public static class JsonClr
{
    public static object? Parse(string json)
    {
        using var doc = JsonDocument.Parse(json);
        return Convert(doc.RootElement);
    }

    /// <summary>Parses a JSON object text into a dictionary; non-object or invalid text gives an empty dictionary.</summary>
    public static IReadOnlyDictionary<string, object?> ParseObject(string? json)
    {
        if (string.IsNullOrWhiteSpace(json)) return new Dictionary<string, object?>();
        try
        {
            if (Parse(json) is JinjaDict d) return d.ToDictionary(kv => (string)kv.Key, kv => kv.Value);
        }
        catch (JsonException) { }
        return new Dictionary<string, object?>();
    }

    public static object? Convert(JsonElement e)
    {
        switch (e.ValueKind)
        {
            case JsonValueKind.Object:
            {
                var d = new Dictionary<string, object?>();
                foreach (var p in e.EnumerateObject()) d[p.Name] = Convert(p.Value);
                return JinjaValue.From(d);
            }
            case JsonValueKind.Array: return e.EnumerateArray().Select(Convert).ToList();
            case JsonValueKind.String: return e.GetString();
            case JsonValueKind.Number: return e.TryGetInt64(out var l) ? l : e.GetDouble();
            case JsonValueKind.True: return true;
            case JsonValueKind.False: return false;
            default: return null;
        }
    }
}
