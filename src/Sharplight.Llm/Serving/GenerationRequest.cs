namespace Sharplight.Llm.Serving;

/// <summary>A token-level generation request.</summary>
public sealed record GenerationRequest
{
    /// <summary>The prompt, already tokenized (not empty, ids inside the vocabulary).</summary>
    public required int[] PromptTokens { get; init; }
    /// <summary>How the next token is chosen.</summary>
    public SamplingParams Sampling { get; init; } = new();
    /// <summary>Maximum generated tokens; 0 or less means as many as the context window allows. Always capped by the room left in the context.</summary>
    public int MaxTokens { get; init; } = 256;
    /// <summary>Extra token ids that end generation (in addition to the end-of-generation tokens when <see cref="StopAtEog"/>).</summary>
    public IReadOnlyList<int> StopTokenIds { get; init; } = [];
    /// <summary>End generation at the model's end-of-generation tokens (false = OpenAI <c>ignore_eos</c>: they are generated like any other token).</summary>
    public bool StopAtEog { get; init; } = true;
    /// <summary>
    /// Minimum generated tokens: until this many exist, end-of-generation tokens and <see cref="StopTokenIds"/> are masked out before
    /// sampling, so they cannot end the generation (stop strings still can). Capped by <see cref="MaxTokens"/> and the context.
    /// </summary>
    public int MinTokens { get; init; }
    /// <summary>Generation ends when one of these appears in the output; the matched string is not emitted.</summary>
    public IReadOnlyList<string> StopStrings { get; init; } = [];
    /// <summary>Higher runs first; within a priority, tenants are served fairly (see <see cref="Tenant"/>), then first-come first-served.</summary>
    public int Priority { get; init; }
    /// <summary>
    /// Fair-share key (a user, an API key, an application). When requests of the same priority wait for a slot, the tenant that
    /// consumed the fewest tokens recently goes first, so one heavy client cannot starve the others. Null = a shared anonymous tenant.
    /// </summary>
    public string? Tenant { get; init; }
    /// <summary>
    /// Restricts the output to a grammar / regular expression / JSON schema (structured outputs). The engine masks the logits
    /// with the tokens the constraint allows before sampling; end-of-generation tokens are allowed only when the output is complete.
    /// </summary>
    public Structured.IOutputConstraint? Constraint { get; init; }
    /// <summary>Total time budget from submission (queue + generation). Exceeding it finishes with <see cref="FinishReason.Timeout"/>.</summary>
    public TimeSpan? Timeout { get; init; }
    /// <summary>Maximum time to wait for a slot; exceeding it finishes with <see cref="FinishReason.Timeout"/> without generating.</summary>
    public TimeSpan? QueueTimeout { get; init; }
    /// <summary>Report the log-probability of every sampled token (and <see cref="TopLogprobs"/> alternatives).</summary>
    public bool Logprobs { get; init; }
    /// <summary>Number of most likely alternatives reported per token (with <see cref="Logprobs"/>).</summary>
    public int TopLogprobs { get; init; }
    /// <summary>Hide control tokens from the generated text.</summary>
    public bool SkipSpecialTokens { get; init; }
}
