namespace Sharplight.Studio.Services.Runtime;

/// <summary>What a <see cref="ChatStreamUpdate"/> carries.</summary>
internal enum ChatStreamKind
{
    /// <summary>A piece of the answer.</summary>
    Content,

    /// <summary>A piece of the model's reasoning.</summary>
    Reasoning,

    /// <summary>A complete tool call (name and JSON arguments as text).</summary>
    ToolCall,

    /// <summary>The end of the answer, with statistics (and an error message when it failed).</summary>
    Finished,
}
