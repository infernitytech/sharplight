namespace Sharplight.Templating;

/// <summary>
/// A Hugging Face style chat template (the <c>tokenizer.chat_template</c> GGUF/HF field). Uses the HF Jinja settings:
/// <c>trim_blocks</c> and <c>lstrip_blocks</c> on, <c>tojson</c> with <c>ensure_ascii=False</c>.
/// </summary>
public sealed class ChatTemplate
{
    private readonly JinjaTemplate _template;
    private readonly string? _bos;
    private readonly string? _eos;

    /// <summary>Compiles <paramref name="source"/>.</summary>
    /// <param name="source">Jinja template text.</param>
    /// <param name="bosToken">Text exposed to the template as <c>bos_token</c>, or null.</param>
    /// <param name="eosToken">Text exposed to the template as <c>eos_token</c>, or null.</param>
    /// <param name="environment">Environment to compile in; defaults to <see cref="CreateDefaultEnvironment"/>.</param>
    /// <exception cref="TemplateSyntaxException">The template does not parse.</exception>
    public ChatTemplate(string source, string? bosToken, string? eosToken, JinjaEnvironment? environment = null)
    {
        Environment = environment ?? CreateDefaultEnvironment();
        _template = Environment.Compile(source);
        _bos = bosToken;
        _eos = eosToken;
    }

    /// <summary>The environment the template was compiled in; use it to register custom filters and tests.</summary>
    public JinjaEnvironment Environment { get; }

    /// <summary>Environment with the HF settings (trim_blocks=True, lstrip_blocks=True).</summary>
    public static JinjaEnvironment CreateDefaultEnvironment() => new() { TrimBlocks = true, LstripBlocks = true };

    /// <summary>
    /// Renders the prompt. Template variables: <c>messages</c>, <c>add_generation_prompt</c>, <c>bos_token</c>, <c>eos_token</c>,
    /// <c>tools</c> (none when null) and everything in <paramref name="extraContext"/> (e.g. <c>enable_thinking</c>).
    /// </summary>
    public string Render(
        IReadOnlyList<ChatMessage> messages,
        bool addGenerationPrompt,
        IReadOnlyDictionary<string, object?>? extraContext = null,
        IReadOnlyList<ToolDefinition>? tools = null)
    {
        var ctx = new Dictionary<string, object?>
        {
            ["bos_token"] = _bos ?? "",
            ["eos_token"] = _eos ?? "",
            ["add_generation_prompt"] = addGenerationPrompt,
        };
        var msgs = new List<object?>(messages.Count);
        foreach (var m in messages)
        {
            msgs.Add(m.ToValue());
        }

        ctx["messages"] = msgs;
        if (tools is { Count: > 0 })
        {
            var tl = new List<object?>(tools.Count);
            foreach (var t in tools)
            {
                tl.Add(t.ToValue());
            }

            ctx["tools"] = tl;
        }
        else
        {
            ctx["tools"] = null;
        }

        if (extraContext != null)
        {
            foreach (var kv in extraContext)
            {
                ctx[kv.Key] = kv.Value;
            }
        }

        return _template.Render(ctx);
    }
}
