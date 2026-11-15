namespace Sharplight.Llm.Structured;

/// <summary>
/// A compiled output constraint (a grammar, a regular expression, a JSON schema) bound to one tokenizer. Immutable and
/// thread-safe: many requests share one instance, each driving its own <see cref="IConstraintMatcher"/>.
/// </summary>
public interface IOutputConstraint
{
    /// <summary>Starts matching a new output from the beginning of the constrained text.</summary>
    public IConstraintMatcher CreateMatcher();
}
