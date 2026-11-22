using System.Collections.Concurrent;

namespace Sharplight.Runtime.Experts;

/// <summary>
/// Reads with positional reads (<c>pread</c>) on a few dedicated threads, one read each at a time: the portable backend, and the Linux fallback
/// where io_uring is unavailable (old kernels, containers whose seccomp profile blocks it). On Linux the files are opened with <c>O_DIRECT</c>.
/// </summary>
internal sealed unsafe class ThreadPoolExpertReader : IExpertReader
{
    private readonly BlockingCollection<(ExpertRead Read, ExpertReadBatch Batch)> _queue = new();
    private readonly Thread[] _threads;

    /// <param name="threads">Reads in flight at most (the queue depth).</param>
    public ThreadPoolExpertReader(int threads)
    {
        _threads = new Thread[Math.Max(1, threads)];
        for (int i = 0; i < _threads.Length; i++)
        {
            _threads[i] = new Thread(Work) { IsBackground = true, Name = $"Sharplight expert reader {i}" };
            _threads[i].Start();
        }
    }

    /// <inheritdoc/>
    public string Name => $"threads x{_threads.Length}";

    /// <inheritdoc/>
    public bool SupportsDeviceDestinations => false;

    /// <inheritdoc/>
    public IExpertFile Open(string path) => ExpertFileHandle.Open(path, direct: true);

    /// <inheritdoc/>
    public ExpertReadBatch Submit(ReadOnlySpan<ExpertRead> reads)
    {
        var batch = new ExpertReadBatch(reads.Length);
        foreach (var read in reads)
        {
            _queue.Add((read, batch));
        }

        return batch;
    }

    private void Work()
    {
        foreach (var (read, batch) in _queue.GetConsumingEnumerable())
        {
            try
            {
                batch.CompleteRead(ReadFully(read));
            }
            catch (Exception e)
            {
                batch.FailRead(e);
            }
        }
    }

    /// <summary>Reads the whole range, or up to the end of the file.</summary>
    private static long ReadFully(ExpertRead read)
    {
        if (read.Destination.HostAddress == 0)
        {
            throw new NotSupportedException("The thread-pool expert reader reads into host memory only.");
        }

        var file = (ExpertFileHandle)read.File;
        long done = 0;
        while (done < read.Length)
        {
            var destination = new Span<byte>((byte*)read.Destination.HostAddress + done, (int)(read.Length - done));
            int n = RandomAccess.Read(file.Handle, destination, read.Offset + done);
            if (n <= 0)
            {
                break;
            }

            done += n;
        }

        return done;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _queue.CompleteAdding();
        foreach (var thread in _threads)
        {
            thread.Join();
        }

        _queue.Dispose();
    }
}
