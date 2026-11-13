namespace Sharplight.Llm.Serving.Chat;

/// <summary>Whether the model may or must call the offered tools.</summary>
public enum ToolChoiceMode
{
    /// <summary>The model decides.</summary>
    Auto,

    /// <summary>Tools are not offered.</summary>
    None,

    /// <summary>The model should call some tool (best effort: expressed to the model through the template).</summary>
    Required,

    /// <summary>Only the function named by <see cref="ChatRequest.ToolChoiceFunction"/> is offered.</summary>
    Function,
}
