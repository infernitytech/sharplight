namespace Sharplight.Runtime.Experts;

/// <summary>
/// Reads parts of model files in batches for <see cref="ExpertStore"/>: the routed experts it leaves on disk. A batch of
/// (file, offset, length, destination) reads is submitted at once and awaited as a whole, so a backend can keep many reads in flight
/// (io_uring on Linux, a pool of threads elsewhere). The destinations are host memory today; a backend that can read into device memory
/// (e.g. DirectStorage on Windows) reports <see cref="SupportsDeviceDestinations"/> and takes <see cref="ExpertReadDestination.DeviceBuffer"/>
/// destinations, without changes to the cache or the placement.
/// </summary>
/// <remarks>Implementations are thread-safe: batches may be submitted from several threads and complete in any order.</remarks>
public interface IExpertReader : IDisposable
{
    /// <summary>Short name of the backend for logs and statistics (e.g. <c>io_uring</c>, <c>threads</c>).</summary>
    public string Name { get; }

    /// <summary>Whether reads may target device memory (<see cref="ExpertReadDestination.DeviceBuffer"/>).</summary>
    public bool SupportsDeviceDestinations { get; }

    /// <summary>Opens a file for reading; the reader may bypass the page cache (<see cref="IExpertFile.Alignment"/> then says how).</summary>
    /// <param name="path">The file.</param>
    /// <returns>The open file; dispose it after the last batch that reads it completed.</returns>
    public IExpertFile Open(string path);

    /// <summary>
    /// Starts the reads. Offsets, lengths and host destinations of a file with an <see cref="IExpertFile.Alignment"/> above 1 must be multiples
    /// of it (the read of a range that ends past the end of the file returns the bytes up to the end).
    /// </summary>
    /// <param name="reads">The reads; the span is not kept.</param>
    /// <returns>The batch, complete when every read finished.</returns>
    public ExpertReadBatch Submit(ReadOnlySpan<ExpertRead> reads);
}
