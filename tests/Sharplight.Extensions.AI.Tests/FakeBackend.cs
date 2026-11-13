using System.Runtime.CompilerServices;

namespace Sharplight.Extensions.AI.Tests;

/// <summary>Scripted backend: records the request and streams the configured pieces.</summary>
internal sealed class FakeBackend(params string[] pieces) : ITextGenerationBackend
{
    public const string Template =
        "{%- if tools %}<|im_start|>system\n# Tools\n{% for t in tools %}{{ t | tojson }}\n{% endfor %}<|im_end|>\n{% endif -%}" +
        "{%- for m in messages %}<|im_start|>{{ m.role }}{{ '\n' }}" +
        "{%- if m.content is string %}{{ m.content }}{% elif m.content %}{% for p in m.content %}{% if p.type == 'image' %}<img>{% else %}{{ p.text }}{% endif %}{% endfor %}{% endif %}" +
        "{%- if m.reasoning_content %}[reasoning:{{ m.reasoning_content }}]{% endif %}" +
        "{%- if m.tool_calls %}[calls:{{ m.tool_calls | length }}:{{ m.tool_calls[0].function.arguments | tojson }}]{% endif %}" +
        "{%- if m.tool_call_id %}[id:{{ m.tool_call_id }}]{% endif %}<|im_end|>\n{% endfor -%}" +
        "{%- if add_generation_prompt %}<|im_start|>assistant\n{% if enable_thinking %}<think>\n{% endif %}{% endif %}";

    public TextGenerationRequest? Last { get; private set; }
    public int PromptTokens { get; set; } = 11;
    public int GeneratedTokens { get; set; } = 5;
    public bool ReachedLimit { get; set; }
    public bool Images { get; set; }

    public string ModelName => "fake";
    public string? ChatTemplate => Template;
    public string? BosToken => null;
    public string? EosToken => null;
    public bool SupportsImages => Images;
    public int ContextLength => 4096;

    public async IAsyncEnumerable<string> GenerateAsync(TextGenerationRequest request, TextGenerationResult result, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        Last = request;
        foreach (var p in pieces)
        {
            cancellationToken.ThrowIfCancellationRequested();
            await Task.Yield();
            yield return p;
        }
        result.PromptTokens = PromptTokens;
        result.GeneratedTokens = GeneratedTokens;
        result.ReachedTokenLimit = ReachedLimit;
    }

    public void Dispose() { }
}
