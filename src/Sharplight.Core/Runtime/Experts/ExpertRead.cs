namespace Sharplight.Runtime.Experts;

/// <summary>One read of a batch (<see cref="IExpertReader.Submit"/>).</summary>
/// <param name="File">The file, opened by the same reader.</param>
/// <param name="Offset">Offset of the first byte in the file.</param>
/// <param name="Length">Bytes to read.</param>
/// <param name="Destination">Where the bytes go.</param>
public readonly record struct ExpertRead(IExpertFile File, long Offset, int Length, ExpertReadDestination Destination);
