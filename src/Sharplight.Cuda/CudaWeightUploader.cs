using Sharplight.Cuda.Interop;
using Sharplight.Runtime;
using Sharplight.Shaders.Lowering;

namespace Sharplight.Cuda;

/// <summary>
/// Streams weight and constant buffers to device memory through up to two host staging buffers: while the GPU copies one, the CPU
/// fills the other (reading mapped files, repacking rows). The copies run on the device's upload stream. Not thread-safe; reusable after
/// <see cref="Wait"/> (the streaming uploader of <see cref="CudaDevice.StreamUploader"/>).
/// </summary>
/// <remarks>
/// The staging buffers are page-locked when possible, at most <see cref="MaxStagingSize"/> and no more than the uploads need. When the
/// system refuses page-locked memory (or <see cref="DeviceMemoryLimits.PinnedHostMemory"/> caps it), the size is halved down to
/// <see cref="MinStagingSize"/>, and below that a pageable buffer is used: the driver then stages each copy itself (slower, still correct).
/// </remarks>
internal sealed unsafe class CudaWeightUploader : IDisposable
{
    private const long MaxStagingSize = 64L << 20;
    private const long MinStagingSize = 1L << 20;
    private const int MaxSlots = 2;

    private readonly CudaDevice _dev;
    private readonly byte*[] _staging = new byte*[MaxSlots];
    private readonly bool[] _pinned = new bool[MaxSlots];
    private readonly nint[] _done = new nint[MaxSlots];
    private readonly int _slots;
    private int _slot;

    /// <summary>Allocates the staging buffers and their events. The device's context must be current.</summary>
    /// <param name="dev">The device.</param>
    /// <param name="totalBytes">Bytes of all buffers to upload (bounds the staging size).</param>
    /// <param name="largestRow">The largest repacked row, which must fit one staging buffer.</param>
    internal CudaWeightUploader(CudaDevice dev, long totalBytes, long largestRow)
    {
        _dev = dev;
        long cap = DeviceMemoryLimits.Apply(MaxStagingSize, dev.Limits.MaxAllocation);
        StagingSize = Math.Max(Math.Max(1, largestRow), Math.Min(cap, Math.Max(MinStagingSize, totalBytes)));
        try
        {
            _staging[0] = AllocatePinnedOrPageable(largestRow, out _pinned[0]);
            _slots = 1;
            // Double buffering only pays for pinned memory (pageable copies are synchronous) and more than one buffer's worth.
            if (_pinned[0] && totalBytes > StagingSize)
            {
                _staging[1] = dev.TryAllocatePinned(StagingSize);
                _pinned[1] = _staging[1] != null;
                _slots = _pinned[1] ? 2 : 1;
            }

            for (int i = 0; i < _slots; i++)
            {
                nint done;
                CudaApi.Check(CudaApi.cuEventCreate(&done, CudaApi.EventDisableTiming), "cuEventCreate");
                _done[i] = done;
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>Bytes per staging buffer.</summary>
    internal long StagingSize { get; private set; }

    /// <summary>Whether the first staging buffer is page-locked.</summary>
    internal bool Pinned => _pinned[0];

    /// <summary>Copies a whole weight or constant buffer to <paramref name="destination"/>, in chunks that fit a staging buffer.</summary>
    internal void Upload(BufferDecl buffer, CudaBuffer destination) => Upload(buffer, destination, 0, buffer.Weight?.ByteSize ?? buffer.InitialData!.Length);

    /// <summary>
    /// Copies bytes [<paramref name="offset"/>, <paramref name="end"/>) of a weight or constant buffer to the same offsets of
    /// <paramref name="destination"/>, in chunks that fit a staging buffer; a repacked weight's range must hold whole repacked rows.
    /// </summary>
    internal void Upload(BufferDecl buffer, CudaBuffer destination, long offset, long end)
    {
        long total = end;
        var source = buffer.Weight?.Source;
        nint pointer = source?.TryGetPointer() ?? 0;
        // Repacked weights are transformed in whole rows.
        long rowBytes = buffer.Repack != WeightRepack.None ? WeightRepacker.RowBytes(buffer.Repack, buffer.RepackRowElements) : 1;
        long chunkBytes = StagingSize - StagingSize % rowBytes;
        if (chunkBytes == 0)
        {
            throw new InvalidOperationException($"A repacked row of {rowBytes} bytes does not fit the {StagingSize}-byte staging buffer.");
        }

        for (long done = offset; done < total; done += chunkBytes)
        {
            long chunk = Math.Min(chunkBytes, total - done);
            // The slot is free once the copy queued from it last time has finished.
            CudaApi.Check(CudaApi.cuEventSynchronize(_done[_slot]), "cuEventSynchronize");
            WeightCopier.CopyChunk(buffer, source, pointer, done, chunk, _staging[_slot], rowBytes);
            if (_pinned[_slot])
            {
                CudaApi.Check(CudaApi.cuMemcpyHtoDAsync(destination.Pointer + (ulong)done, _staging[_slot], (nuint)chunk, _dev.Stream), "cuMemcpyHtoDAsync");
            }
            else
            {
                // Pageable memory: on the upload stream after the zero fill (never on the legacy stream, which waits for every blocking
                // stream of the context and fails while another thread captures a graph), then waited for, so the slot is free again.
                CudaApi.Check(CudaApi.cuMemcpyHtoDAsync(destination.Pointer + (ulong)done, _staging[_slot], (nuint)chunk, _dev.Stream), "cuMemcpyHtoDAsync");
                CudaApi.Check(CudaApi.cuStreamSynchronize(_dev.Stream), "cuStreamSynchronize");
            }

            CudaApi.Check(CudaApi.cuEventRecord(_done[_slot], _dev.Stream), "cuEventRecord");
            _slot = (_slot + 1) % _slots;
        }
    }

    /// <summary>
    /// A page-locked buffer of <see cref="StagingSize"/> bytes, halving the size on failure; a pageable one of the full size when even
    /// the smallest page-locked buffer is refused.
    /// </summary>
    private byte* AllocatePinnedOrPageable(long largestRow, out bool pinned)
    {
        long minimum = Math.Max(MinStagingSize, largestRow);
        for (long size = StagingSize; size >= minimum; size /= 2)
        {
            byte* memory = _dev.TryAllocatePinned(size);
            if (memory != null)
            {
                StagingSize = size;
                pinned = true;
                return memory;
            }
        }

        pinned = false;
        return (byte*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)StagingSize, 4096);
    }

    /// <summary>Waits for the queued copies; the uploader can be used again.</summary>
    internal void Wait() => CudaApi.Check(CudaApi.cuStreamSynchronize(_dev.Stream), "cuStreamSynchronize");

    /// <summary>Frees the staging buffers and events after waiting for the queued copies.</summary>
    public void Dispose()
    {
        CudaApi.cuStreamSynchronize(_dev.Stream);
        for (int i = 0; i < MaxSlots; i++)
        {
            if (_staging[i] != null)
            {
                _dev.FreeHost(_staging[i], StagingSize, _pinned[i]);
                _staging[i] = null;
            }

            if (_done[i] != 0)
            {
                CudaApi.cuEventDestroy(_done[i]);
                _done[i] = 0;
            }
        }
    }
}
