namespace Sharplight.Import.PyTorch;

/// <summary>
/// A checkpoint's pickle stream uses something the restricted unpickler refuses: an opcode or a global outside the tensor
/// state-dict subset, or a structure that violates a size limit. Nothing from the stream has been executed.
/// </summary>
public sealed class UnsafePickleException : Exception
{
    /// <summary>Creates the exception with <paramref name="message"/>.</summary>
    /// <param name="message">What was refused and why.</param>
    public UnsafePickleException(string message)
        : base(message)
    {
    }
}
