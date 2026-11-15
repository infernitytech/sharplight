using Sharplight.Llm.Structured;

namespace Sharplight.Llm.Serving.Chat;

/// <summary>
/// A chat completion request for <see cref="ModelService.StartChat"/>: the conversation, optional tools and the generation
/// settings. The messages are rendered with the model's chat template.
/// </summary>
public sealed class ChatRequest
{
    /// <summary>The conversation so far, oldest first.</summary>
    public required IReadOnlyList<ChatTurn> Messages { get; init; }
    /// <summary>Functions the model may call; ignored when <see cref="ToolChoice"/> is <see cref="ToolChoiceMode.None"/>.</summary>
    public IReadOnlyList<ToolSpec> Tools { get; init; } = [];
    /// <summary>Whether and which tools the model should use.</summary>
    public ToolChoiceMode ToolChoice { get; init; }
    /// <summary>Function name for <see cref="ToolChoiceMode.Function"/>.</summary>
    public string? ToolChoiceFunction { get; init; }
    /// <summary>How the next token is chosen.</summary>
    public SamplingParams Sampling { get; init; } = new();
    /// <summary>0 = as many as fit into the context.</summary>
    public int MaxTokens { get; init; }
    /// <summary>Generation ends when one of these appears in the content (it is cut off).</summary>
    public IReadOnlyList<string> StopStrings { get; init; } = [];
    /// <summary>Extra token ids that end generation.</summary>
    public IReadOnlyList<int> StopTokenIds { get; init; } = [];
    /// <summary>Keep generating through end-of-generation tokens (OpenAI <c>ignore_eos</c>, for benchmarks).</summary>
    public bool IgnoreEos { get; init; }
    /// <summary>Generated tokens before end-of-generation or stop tokens may end the output (see <see cref="GenerationRequest.MinTokens"/>).</summary>
    public int MinTokens { get; init; }
    /// <summary>Denoising schedule of masked-diffusion models (see <see cref="GenerationRequest.Diffusion"/>); ignored by autoregressive models.</summary>
    public DiffusionOptions? Diffusion { get; init; }
    /// <summary>Number of independent completions (each is its own engine request sharing the prompt prefix).</summary>
    public int N { get; init; } = 1;
    /// <summary>Higher runs first when requests queue for a slot.</summary>
    public int Priority { get; init; }
    /// <summary>Who the request is charged to for fair queuing among equal priorities (see <see cref="GenerationRequest.Tenant"/>).</summary>
    public string? Tenant { get; init; }
    /// <summary>Total time budget (queue plus generation); null uses the engine default.</summary>
    public TimeSpan? Timeout { get; init; }
    /// <summary>Passed to the chat template as <c>enable_thinking</c> when set.</summary>
    public bool? EnableThinking { get; init; }
    /// <summary>JSON mode: instructs the model to answer with a JSON object only and enforces it (as <see cref="OutputFormat.JsonObject"/>) unless <see cref="ResponseFormat"/> is set.</summary>
    public bool JsonMode { get; init; }
    /// <summary>
    /// Structure the answer must have, enforced with constrained decoding (OpenAI <c>response_format</c>). A reasoning block
    /// before it stays free; with tools the model may answer in this format or call a tool.
    /// </summary>
    public OutputFormat? ResponseFormat { get; init; }
    /// <summary>Report the log-probability of every sampled token.</summary>
    public bool Logprobs { get; init; }
    /// <summary>Alternatives reported per token (with <see cref="Logprobs"/>).</summary>
    public int TopLogprobs { get; init; }
    /// <summary>Extra variables for the chat template (<c>chat_template_kwargs</c>).</summary>
    public IReadOnlyDictionary<string, object?>? TemplateContext { get; init; }
}
