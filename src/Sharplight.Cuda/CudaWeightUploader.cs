using Sharplight.Cuda.Interop;
using Sharplight.Shaders.Lowering;

namespace Sharplight.Cuda;

/// <summary>
/// Streams weight and constant buffers to device memory through two pinned host staging buffers: while the GPU copies one, the CPU fills
/// the other (reading mapped files, repacking rows). The copies run on the device's upload stream. Single use, not thread-safe.
/// </summary>
internal sealed unsafe class CudaWeightUploader : IDisposable
{
    private const long StagingSize = 64L << 20;
    private const int Slots = 2;

    private readonly CudaDevice _dev;
    private readonly byte*[] _staging = new byte*[Slots];
    private readonly nint[] _done = new nint[Slots];
    private int _slot;

    /// <summary>Allocates the pinned staging buffers and their events. The device's context must be current.</summary>
    internal CudaWeightUploader(CudaDevice dev)
    {
        _dev = dev;
        try
        {
            for (int i = 0; i < Slots; i++)
            {
                void* staging;
                CudaApi.Check(CudaApi.cuMemAllocHost(&staging, (nuint)StagingSize), "cuMemAllocHost");
                _staging[i] = (byte*)staging;
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

    /// <summary>Copies a whole weight or constant buffer to <paramref name="destination"/>, in chunks that fit a staging buffer.</summary>
    internal void Upload(BufferDecl buffer, CudaBuffer destination)
    {
        long total = buffer.Weight?.ByteSize ?? buffer.InitialData!.Length;
        var source = buffer.Weight?.Source;
        nint pointer = source?.TryGetPointer() ?? 0;
        // Repacked weights are transformed in whole rows.
        long rowBytes = buffer.Repack != WeightRepack.None ? WeightRepacker.RowBytes(buffer.Repack, buffer.RepackRowElements) : 1;
        long chunkBytes = StagingSize - StagingSize % rowBytes;
        if (chunkBytes == 0)
        {
            throw new InvalidOperationException($"A repacked row of {rowBytes} bytes does not fit the {StagingSize}-byte staging buffer.");
        }

        for (long done = 0; done < total; done += chunkBytes)
        {
            long chunk = Math.Min(chunkBytes, total - done);
            // The slot is free once the copy queued from it last time has finished.
            CudaApi.Check(CudaApi.cuEventSynchronize(_done[_slot]), "cuEventSynchronize");
            WeightCopier.CopyChunk(buffer, source, pointer, done, chunk, _staging[_slot], rowBytes);
            CudaApi.Check(CudaApi.cuMemcpyHtoDAsync(destination.Pointer + (ulong)done, _staging[_slot], (nuint)chunk, _dev.Stream), "cuMemcpyHtoDAsync");
            CudaApi.Check(CudaApi.cuEventRecord(_done[_slot], _dev.Stream), "cuEventRecord");
            _slot ^= 1;
        }
    }

    /// <summary>Frees the staging buffers and events after waiting for the queued copies.</summary>
    public void Dispose()
    {
        CudaApi.cuStreamSynchronize(_dev.Stream);
        for (int i = 0; i < Slots; i++)
        {
            if (_staging[i] != null)
            {
                CudaApi.cuMemFreeHost(_staging[i]);
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
