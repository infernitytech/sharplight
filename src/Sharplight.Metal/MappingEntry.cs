namespace Sharplight.Metal;

/// <summary>One wrapped file mapping of a <see cref="MetalDevice"/>: its buffer, the page-aligned start and the views using it.</summary>
/// <param name="handle">The retained <c>MTLBuffer</c> over the mapping's pages.</param>
/// <param name="start">The page-aligned address the buffer starts at.</param>
internal sealed class MappingEntry(nint handle, nint start)
{
    public nint Handle { get; } = handle;

    public nint Start { get; } = start;

    /// <summary>Weight buffers addressing the mapping (guarded by the device's mapping lock).</summary>
    public int References { get; set; }
}
