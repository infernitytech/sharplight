namespace Sharplight.Llm.Serving;

/// <summary>Media content placed in a prompt: its rows fill prompt positions <see cref="Start"/> .. <see cref="Start"/> + <see cref="MediaContent.Rows"/> - 1.</summary>
/// <param name="Start">First prompt position the content's rows fill.</param>
/// <param name="Content">The content.</param>
public readonly record struct PromptMedia(int Start, MediaContent Content)
{
    /// <summary>The position after the content's last row.</summary>
    public int End => Start + Content.Rows;
}
