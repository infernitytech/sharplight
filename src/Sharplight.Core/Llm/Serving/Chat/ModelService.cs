using System.Text;
using System.Text.Json;
using Sharplight.Llm.Structured;
using Sharplight.Templating;
using Sharplight.Tokenizers;

namespace Sharplight.Llm.Serving.Chat;

/// <summary>
/// The chat layer on top of an <see cref="InferenceEngine"/>: renders messages (with tools) through the model's Jinja chat
/// template, submits the requests and turns the generated text into content / reasoning / tool-call events.
/// Thread-safe; many requests may be started concurrently (the engine batches them).
/// </summary>
/// <example>
/// <code>
/// var service = new ModelService(engine, model.ChatTemplate, new ModelServiceOptions { ModelName = "qwen3" });
/// var request = new ChatRequest
/// {
///     Messages = [new ChatTurn("user", "Name three primary colours.")],
///     MaxTokens = 128,
/// };
/// var result = await service.StartChat(request).CompleteAsync();
/// Console.WriteLine(result.Choices[0].Content);
/// </code>
/// </example>
public sealed class ModelService
{
    private readonly ChatTemplate? _template;
    private readonly IReadOnlyList<IToolCallFormat> _formats;
    private readonly string? _bos;
    /// <summary>The template can emit a <c>&lt;think&gt;</c> block (structured output leaves it free).</summary>
    private readonly bool _reasoningTemplate;
    private readonly bool _harmony;   // OpenAI harmony output (gpt-oss): channels and recipients instead of think / tool markup
    /// <summary>Compiled output constraints by request shape (format, tools, flags); cleared when it reaches <see cref="MaxCachedConstraints"/>.</summary>
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, GrammarConstraint> _constraints = new(StringComparer.Ordinal);
    private const int MaxCachedConstraints = 64;

    /// <summary>The engine requests are submitted to.</summary>
    public InferenceEngine Engine { get; }
    /// <summary>The service settings.</summary>
    public ModelServiceOptions Options { get; }
    /// <summary>The tokenizer of the served model.</summary>
    public ITokenizer Tokenizer => Engine.Tokenizer;
    /// <summary>Name reported by the APIs.</summary>
    public string ModelName => Options.ModelName;
    /// <summary>False when the model has no chat template and a plain <c>Role: text</c> transcript is used.</summary>
    public bool HasChatTemplate => _template is not null;

    /// <summary>Creates the chat layer over <paramref name="engine"/>; the caller keeps ownership of the engine.</summary>
    /// <param name="engine">The scheduler that runs the generations.</param>
    /// <param name="chatTemplate">The model's Jinja chat template (<see cref="LlmModel.ChatTemplate"/>); <see cref="ModelServiceOptions.ChatTemplate"/> overrides it.</param>
    /// <param name="options">Name and defaults; library defaults when null.</param>
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

        _harmony = src?.Contains("<|channel|>", StringComparison.Ordinal) == true;
        _formats = Options.ToolFormats ?? (_harmony ? [ToolCallFormats.Hermes] : ToolCallFormats.ForTemplate(src));
        _reasoningTemplate = src?.Contains("<think>", StringComparison.Ordinal) == true;
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
        if (r.JsonMode || r.ResponseFormat?.IsJson == true)
        {
            string hint = r.ResponseFormat?.Schema is { } schema
                ? "Respond only with valid JSON that matches this JSON schema, and no other text:\n" + schema
                : "Respond only with a single valid JSON object and no other text.";
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

    /// <summary>Number of prompt tokens the request would occupy after template rendering.</summary>
    /// <exception cref="EngineException">The request contains images (not supported by this configuration).</exception>
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
    /// <param name="r">The conversation and settings.</param>
    /// <param name="ct">Cancels all choices.</param>
    /// <exception cref="EngineException">The request was not admitted (see <see cref="InferenceEngine.Submit"/>).</exception>
    public ChatRun StartChat(ChatRequest r, CancellationToken ct = default)
    {
        var prompt = BuildPrompt(r);
        bool useTools = r.Tools.Count > 0 && r.ToolChoice != ToolChoiceMode.None;
        var names = r.Tools.Select(t => t.Name).ToList();
        var template = new GenerationRequest
        {
            PromptTokens = prompt.Tokens, Sampling = r.Sampling, MaxTokens = r.MaxTokens, MinTokens = r.MinTokens, StopAtEog = !r.IgnoreEos,
            StopStrings = r.StopStrings, StopTokenIds = r.StopTokenIds, Priority = r.Priority, Tenant = r.Tenant, Timeout = r.Timeout,
            Logprobs = r.Logprobs, TopLogprobs = r.TopLogprobs, Constraint = BuildConstraint(r, prompt), Diffusion = r.Diffusion,
        };
        return Start(template, r.N, () => new ChatOutputParser(useTools ? _formats : null, useTools ? names : null, prompt.StartsInReasoning, harmony: _harmony), ct);
    }

    /// <summary>Submits a raw text completion request.</summary>
    /// <exception cref="EngineException">The request was not admitted (see <see cref="InferenceEngine.Submit"/>).</exception>
    public ChatRun StartCompletion(CompletionRequest r, CancellationToken ct = default)
    {
        var tokens = r.PromptTokens ?? Tokenizer.Encode(r.Prompt ?? "", addSpecial: true, parseSpecial: true);
        var template = new GenerationRequest
        {
            PromptTokens = tokens, Sampling = r.Sampling, MaxTokens = r.MaxTokens, MinTokens = r.MinTokens, StopAtEog = !r.IgnoreEos,
            StopStrings = r.StopStrings, StopTokenIds = r.StopTokenIds, Priority = r.Priority, Tenant = r.Tenant, Timeout = r.Timeout,
            Logprobs = r.Logprobs, TopLogprobs = r.TopLogprobs, Diffusion = r.Diffusion,
            Constraint = r.ResponseFormat is { } format ? Compile("completion|" + format.CacheKey, b => format.AddTo(b), null) : null,
        };
        return Start(template, r.N, null, ct);
    }

    // ---- structured output ------------------------------------------------------------------------------------------

    /// <summary>
    /// The output constraint of a chat request, or null when it has none: the response format and/or the tool-call syntax
    /// with each tool's parameter schema, after an optional (free) reasoning block. Forced tool calls are always
    /// constrained; with <see cref="ToolChoiceMode.Auto"/> and strict tools the grammar starts at the format's start marker.
    /// </summary>
    internal IOutputConstraint? BuildConstraint(ChatRequest r, ChatPrompt prompt)
    {
        bool useTools = r.Tools.Count > 0 && r.ToolChoice != ToolChoiceMode.None;
        var tools = useTools && r.ToolChoice == ToolChoiceMode.Function ? r.Tools.Where(t => t.Name == r.ToolChoiceFunction).ToList() : r.Tools;
        if (tools.Count == 0)
        {
            tools = r.Tools;
        }

        bool forced = useTools && r.ToolChoice is ToolChoiceMode.Required or ToolChoiceMode.Function;
        bool strict = useTools && r.ToolChoice == ToolChoiceMode.Auto && r.Tools.Any(t => t.Strict);
        var format = r.ResponseFormat ?? (r.JsonMode ? OutputFormat.JsonObject : null);
        // Only a template-specific format tells which syntax to enforce (an unknown template lists all of them).
        var callFormat = _formats.Count == 1 ? _formats[0] : null;
        if (format is null && !(callFormat is not null && (forced || strict)))
        {
            return null;
        }

        if (format is null && !forced)
        {
            // Strict tools, free answer: the grammar takes over at the call's start marker.
            if (callFormat!.StartMarker is not { } marker)
            {
                return null;
            }

            return Compile("strict|" + ToolsKey(callFormat, tools), b => ToolCallGrammar.Calls(b, callFormat, tools, afterStartMarker: true)!,
                new GrammarConstraintOptions { Trigger = marker, TriggerOptional = true, SpecialTokens = ToolCallGrammar.Markers });
        }

        string key = $"chat|{format?.CacheKey}|{forced}|{strict}|{prompt.StartsInReasoning}|{(callFormat is not null && (forced || strict) ? ToolsKey(callFormat, tools) : "")}";
        return Compile(key, b =>
        {
            var calls = callFormat is not null && (forced || strict) ? ToolCallGrammar.Calls(b, callFormat, tools, afterStartMarker: false) : null;
            var body = forced && calls is not null ? calls
                : calls is not null ? GrammarExpr.Choice(format!.AddTo(b), calls)
                : format!.AddTo(b);
            if (prompt.StartsInReasoning)
            {
                return GrammarExpr.Seq(b.Until("</think>"), JsonSchemaCompiler.Whitespace(), body);
            }

            if (_reasoningTemplate)
            {
                var think = GrammarExpr.Seq(GrammarExpr.Literal("<think>"), b.Until("</think>"), JsonSchemaCompiler.Whitespace());
                return GrammarExpr.Seq(GrammarExpr.Optional(think), body);
            }

            return body;
        }, new GrammarConstraintOptions { SpecialTokens = ToolCallGrammar.Markers });
    }

    private static string ToolsKey(IToolCallFormat format, IReadOnlyList<ToolSpec> tools)
        => format.Name + string.Concat(tools.Select(t => $"|{t.Name}\u0001{t.ParametersJson}"));

    /// <summary>
    /// The constraint for the grammar built by <paramref name="root"/>, bound to the tokenizer. Constraints are cached by
    /// <paramref name="key"/>: a repeated schema or tool set keeps its compiled grammar and its shared token-mask cache.
    /// Invalid formats are request errors.
    /// </summary>
    private GrammarConstraint Compile(string key, Func<GrammarBuilder, GrammarExpr> root, GrammarConstraintOptions? options)
    {
        if (_constraints.TryGetValue(key, out var cached))
        {
            return cached;
        }

        try
        {
            var builder = new GrammarBuilder();
            JsonSchemaCompiler.DefineCommonRules(builder);
            builder.Add("root", root(builder));
            var constraint = new GrammarConstraint(builder.Build("root"), Tokenizer, options);
            if (_constraints.Count >= MaxCachedConstraints)
            {
                _constraints.Clear();
            }

            _constraints[key] = constraint;
            return constraint;
        }
        catch (Exception e) when (e is FormatException or ArgumentException or NotSupportedException or JsonException)
        {
            throw new EngineException("Invalid structured output format: " + e.Message, "invalid_request");
        }
    }

    /// <summary>Submits <paramref name="n"/> copies of <paramref name="template"/> (distinct seeds when seeded).</summary>
    private ChatRun Start(GenerationRequest template, int n, Func<ChatOutputParser>? parser, CancellationToken ct)
    {
        n = Math.Max(1, n);
        int maxTokens = template.MaxTokens > 0 ? template.MaxTokens : Options.DefaultMaxTokens;
        var handles = new List<GenerationHandle>(n);
        try
        {
            for (int i = 0; i < n; i++)
            {
                var sampling = template.Sampling.Seed is int s && i > 0 ? template.Sampling with { Seed = s + i } : template.Sampling;
                handles.Add(Engine.Submit(template with { Sampling = sampling, MaxTokens = maxTokens }, ct));
            }
        }
        catch
        {
            foreach (var h in handles)
            {
                h.Cancel();
            }

            throw;
        }
        return new ChatRun(handles, parser, template.PromptTokens.Length);
    }
}
