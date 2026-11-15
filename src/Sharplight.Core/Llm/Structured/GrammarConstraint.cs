using Sharplight.Tokenizers;

namespace Sharplight.Llm.Structured;

/// <summary>
/// Constrains generation to a <see cref="Grammar"/>: before every token the allowed tokens are computed by walking the
/// tokenizer's precompiled vocabulary trie in step with the grammar, and the engine masks the logits with them (on the CPU,
/// or handed to a GPU sampler as a bitmask). Immutable and thread-safe; share one instance between requests.
/// </summary>
public sealed class GrammarConstraint : IOutputConstraint
{
    private readonly TokenTrie _trie;
    private readonly int[] _specialTokens;
    private readonly byte[]? _trigger;
    private readonly TokenMaskCache _cache;

    /// <summary>Binds <paramref name="grammar"/> to <paramref name="tokenizer"/>.</summary>
    /// <param name="grammar">The grammar the output (after the trigger, if any) must match.</param>
    /// <param name="tokenizer">The tokenizer of the model; its vocabulary trie is built on first use and cached.</param>
    /// <param name="options">Trigger and special-token settings; defaults when null.</param>
    public GrammarConstraint(Grammar grammar, ITokenizer tokenizer, GrammarConstraintOptions? options = null)
    {
        Grammar = grammar;
        Tokenizer = tokenizer;
        Options = options ?? new();
        _trie = TokenTrie.For(tokenizer);
        _specialTokens = [.. Options.SpecialTokens.Select(t => _trie.SpecialTokens.TryGetValue(t, out int id) ? id : -1).Where(id => id >= 0).Distinct()];
        _trigger = string.IsNullOrEmpty(Options.Trigger) ? null : System.Text.Encoding.UTF8.GetBytes(Options.Trigger);
        _cache = new TokenMaskCache(grammar, _trie);
    }

    /// <summary>The grammar.</summary>
    public Grammar Grammar { get; }
    /// <summary>The tokenizer the token masks are computed for.</summary>
    public ITokenizer Tokenizer { get; }
    /// <summary>The settings.</summary>
    public GrammarConstraintOptions Options { get; }

    /// <inheritdoc/>
    public IConstraintMatcher CreateMatcher() => new GrammarMatcher(this, _trie, _cache, _specialTokens, _trigger);
}
