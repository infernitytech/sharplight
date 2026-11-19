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
    /// Isolation scope of the prefix cache: a request only reuses (or shares KV blocks of) contexts left by requests with the same
    /// partition, so identical prompts of different scopes (tenants, spaces) never share cache entries, and a cache hit cannot
    /// reveal across scopes that another scope sent the same prefix (timing side channel). Null is one partition of its own (the
    /// default, single-scope behaviour).
    /// </summary>
    public string? CachePartition { get; init; }
    /// <summary>
    /// Additional prefix-cache key (vLLM's <c>cache_salt</c>): mixed into the hash of every cached KV block together with
    /// <see cref="CachePartition"/>, so only requests with the same partition and salt share cached prefixes. Null = no salt.
    /// Unlike the partition, which a host derives from authentication, a salt may come from the client (it can only narrow sharing
    /// further within the partition).
    /// </summary>
    public string? CacheSalt { get; init; }
    /// <summary>
    /// Non-text content in the prompt (image or audio placeholders): its hashes are part of the prefix-cache key of the blocks it
    /// overlaps, so prompts with the same placeholder tokens but different content never share them.
    /// </summary>
    public IReadOnlyList<PromptContentKey> PromptContent { get; init; } = [];
    /// <summary>
    /// Image or audio content whose embedding rows fill prompt positions in place of their placeholder tokens (non-overlapping). The engine
    /// encodes each content once, right before the prompt chunk that needs it (cached by hash, so a repeated image is not encoded again), and
    /// its hash is part of the prefix-cache key like <see cref="PromptContent"/>. Needs a decoder created with media input
    /// (<see cref="IBatchDecoder.EmbeddingWidth"/> &gt; 0). Build image prompts with <see cref="ImagePrompts"/>.
    /// </summary>
    public IReadOnlyList<PromptMedia> PromptMedia { get; init; } = [];
    /// <summary>
    /// Rope position ids of the prompt rows, [prompt length, <see cref="IBatchDecoder.PositionAxes"/>] row-major, for models whose image
    /// spans advance the position by less than their rows (M-RoPE); null: each row's position is its index.
    /// </summary>
    public int[]? PromptPositions { get; init; }
    /// <summary>
    /// Rope position of the first generated token (the row after the prompt) when <see cref="PromptPositions"/> is given; null: the prompt
    /// length. Every later row continues from it.
    /// </summary>
    public int? NextPosition { get; init; }
    /// <summary>
    /// Per prompt row, whether it goes through the language model's media weight set (CogVLM's visual experts); null: the
    /// <see cref="PromptMedia"/> rows do and no other row.
    /// </summary>
    public bool[]? PromptMediaWeights { get; init; }
    /// <summary>
    /// Compute the whole prompt instead of reusing cached KV (the request still leaves its blocks for later ones). Set it for requests
    /// that need every prompt position computed, e.g. prompt log-probabilities.
    /// </summary>
    public bool BypassPrefixCache { get; init; }
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
    /// <summary>
    /// Masked-diffusion models (<see cref="DiffusionBatchDecoder"/>): the denoising schedule of the generated span of <see cref="MaxTokens"/>
    /// tokens (null: the defaults of <see cref="DiffusionOptions"/>). Ignored by autoregressive models.
    /// </summary>
    public DiffusionOptions? Diffusion { get; init; }
}
