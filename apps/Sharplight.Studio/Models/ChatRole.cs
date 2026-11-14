namespace Sharplight.Studio.Models;

/// <summary>Author of a chat message.</summary>
internal enum ChatRole
{
    /// <summary>Instructions for the model.</summary>
    System,

    /// <summary>The person chatting.</summary>
    User,

    /// <summary>The model.</summary>
    Assistant,
}
