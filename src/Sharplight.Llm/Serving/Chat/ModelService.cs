using System.Text;
using Sharplight.Templating;
using Sharplight.Tokenizers;

namespace Sharplight.Llm.Serving.Chat;

/// <summary>
/// The chat layer on top of an <see cref="InferenceEngine"/>: renders messages (with tools) through the model's Jinja chat
/// template, submits the requests and turns the generated text into content / reasoning / tool-call events.
/// </summary>
public sealed class ModelService
{
    private readonly ChatTemplate? _template;
    private readonly IReadOnlyList<IToolCallFormat> _formats;
    private readonly string? _bos;

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
        if (!string.IsNullOrWhiteSpace(src))
        {
            _template = new ChatTemplate(src, _bos, Tok(engine.Tokenizer.EosId));
        }

        _formats = Options.ToolFormats ?? ToolCallFormats.ForTemplate(src);
    }

    // ---- prompt -----------------------------------------------------------------------------------------------------

    internal ChatPrompt BuildPrompt(ChatRequest r)
    {
        foreach (var m in r.Messages)
        {
            if (m.Parts?.Any(p => p.Type == "image") == true)
            {
                throw new EngineException("Image inputs are not supported by this serving configuration.", "invalid_request");
            }
        }

        bool useTools = r.Tools.Count > 0 && r.ToolChoice != ToolChoiceMode.None;
        var tools = useTools ? r.Tools.Where(t => r.ToolChoice != ToolChoiceMode.Function || t.Name == r.ToolChoiceFunction).Select(ToDefinition).ToList() : null;
        if (useTools && tools!.Count == 0)
        {
            tools = [.. r.Tools.Select(ToDefinition)];
        }

        var msgs = r.Messages.Select(ToMessage).ToList();
        if (r.JsonMode)
        {
            const string hint = "Respond only with a single valid JSON object and no other text.";
            int sys = msgs.FindIndex(m => m.Role == "system");
            if (sys >= 0)
            {
                msgs[sys] = msgs[sys] with { Content = (msgs[sys].Content ?? "") + "\n\n" + hint };
            }
            else
            {
                msgs.Insert(0, new Templating.ChatMessage("system", hint));
            }
        }
        Dictionary<string, object?>? ctx = null;
        if (r.EnableThinking is bool et)
        {
            (ctx ??= [])["enable_thinking"] = et;
        }

        if (r.TemplateContext is not null)
        {
            foreach (var kv in r.TemplateContext)
            {
                (ctx ??= [])[kv.Key] = kv.Value;
            }
        }

        string text = _template is not null
            ? _template.Render(msgs, addGenerationPrompt: true, ctx, tools)
            : FallbackPrompt(msgs);
        bool addSpecial = Tokenizer.AddBos && _bos is not null && !text.StartsWith(_bos, StringComparison.Ordinal);
        var tokens = Tokenizer.Encode(text, addSpecial, parseSpecial: true);
        return new ChatPrompt(text, tokens, text.TrimEnd().EndsWith("<think>", StringComparison.Ordinal));
    }

    public int CountTokens(ChatRequest r) => BuildPrompt(r).Tokens.Length;

    private static string FallbackPrompt(List<Templating.ChatMessage> msgs)
    {
        var sb = new StringBuilder();
        foreach (var m in msgs)
        {
            sb.Append(char.ToUpperInvariant(m.Role[0])).Append(m.Role[1..]).Append(": ").Append(m.Content).Append('\n');
        }

        return sb.Append("Assistant:").ToString();
    }

    private static ToolDefinition ToDefinition(ToolSpec t)
    {
        IReadOnlyDictionary<string, object?>? pars = null;
        if (!string.IsNullOrWhiteSpace(t.ParametersJson) && JsonClr.Parse(t.ParametersJson) is JinjaDict d)
        {
            pars = d.ToDictionary(kv => (string)kv.Key, kv => kv.Value);
        }

        return new ToolDefinition(t.Name, t.Description, pars);
    }

    private static Templating.ChatMessage ToMessage(ChatTurn t)
    {
        string? content = t.Content;
        IReadOnlyList<Templating.ChatContentPart>? parts = null;
        if (t.Parts is { Count: > 0 } ps)
        {
            if (ps.All(p => p.Type == "text"))
            {
                content = string.Concat(ps.Select(p => p.Text));
            }
            else
            {
                parts = [.. ps.Select(p => p.Type == "text" ? Templating.ChatContentPart.FromText(p.Text ?? "") : Templating.ChatContentPart.Image())];
            }
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

    private ChatRun Start(int[] tokens, int n, SamplingParams sampling, int maxTokens, IReadOnlyList<string> stops, IReadOnlyList<int> stopIds, int priority,
        TimeSpan? timeout, bool logprobs, int topLogprobs, Func<ChatOutputParser>? parser, CancellationToken ct)
    {
        n = Math.Max(1, n);
        var handles = new List<GenerationHandle>(n);
        try
        {
            for (int i = 0; i < n; i++)
            {
                handles.Add(Engine.Submit(new GenerationRequest
                {
                    PromptTokens = tokens,
                    Sampling = sampling.Seed is int s && i > 0 ? sampling with { Seed = s + i } : sampling,
                    MaxTokens = maxTokens > 0 ? maxTokens : Options.DefaultMaxTokens,
                    StopStrings = stops, StopTokenIds = stopIds, Priority = priority, Timeout = timeout,
                    Logprobs = logprobs, TopLogprobs = topLogprobs,
                }, ct));
            }
        }
        catch
        {
            foreach (var h in handles) { h.Cancel(); }
            throw;
        }
        return new ChatRun(handles, parser, tokens.Length);
    }
}
