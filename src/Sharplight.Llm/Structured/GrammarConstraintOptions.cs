namespace Sharplight.Llm.Structured;

/// <summary>Optional behaviour of a <see cref="GrammarConstraint"/>.</summary>
public sealed class GrammarConstraintOptions
{
    /// <summary>
    /// When set, the output is unconstrained until this text appears (for example <c>&lt;/think&gt;</c> after a reasoning
    /// block, or a tool-call start marker); the grammar then applies to the text after it.
    /// </summary>
    public string? Trigger { get; init; }

    /// <summary>
    /// With a <see cref="Trigger"/>: whether the output may end without the trigger ever appearing (a tool call the model may
    /// or may not make). When false the end-of-generation tokens stay blocked until the grammar is complete.
    /// </summary>
    public bool TriggerOptional { get; init; }

    /// <summary>
    /// Special (control) tokens the grammar may produce by their text, such as a tool-call marker that is a single control
    /// token. Other special tokens are never allowed while the grammar is active.
    /// </summary>
    public IReadOnlyList<string> SpecialTokens { get; init; } = [];

    /// <summary>Allowed-token masks cached per grammar state and request (each costs VocabSize/8 bytes).</summary>
    public int MaskCacheSize { get; init; } = 64;
}
