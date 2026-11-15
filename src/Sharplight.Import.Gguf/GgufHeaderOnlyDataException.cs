namespace Sharplight.Import.Gguf;

/// <summary>
/// A step of the import read tensor data while the file was checked from its header alone (<see cref="GgufSupport.Validate"/>), so that
/// step could not be checked.
/// </summary>
internal sealed class GgufHeaderOnlyDataException : InvalidOperationException
{
    /// <summary>Creates the exception for the tensor whose data was read.</summary>
    /// <param name="tensor">The tensor's GGUF name.</param>
    public GgufHeaderOnlyDataException(string tensor)
        : base($"The data of tensor '{tensor}' is not available: only the file's header was read.")
    {
    }
}
