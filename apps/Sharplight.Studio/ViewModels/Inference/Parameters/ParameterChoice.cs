namespace Sharplight.Studio.ViewModels.Inference.Parameters;

/// <summary>One option of a <see cref="ChoiceParameterViewModel"/>.</summary>
/// <param name="Value">What the page maps to the options (empty for "automatic").</param>
/// <param name="Label">Shown in the list.</param>
internal sealed record ParameterChoice(string Value, string Label)
{
    /// <inheritdoc/>
    public override string ToString() => Label;
}
