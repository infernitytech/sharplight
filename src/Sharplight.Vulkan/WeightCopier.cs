using System.Buffers;
using Sharplight.Models;
using Sharplight.Shaders.Lowering;

namespace Sharplight.Vulkan;

/// <summary>Host-side copying of weight data into staging memory: parallel copies, row repacking and background materialization.</summary>
internal static unsafe class WeightCopier
{
    /// <summary>Bytes copied per parallel piece (weight sources are usually mmapped files).</summary>
    private const long CopyPieceBytes = 2 << 20;

    /// <summary>Rows repacked per parallel task.</summary>
    private const long RepackRowsPerTask = 256;

    /// <summary>
    /// Produces the weights that are computed on first read (layout conversions, folded constants) on several threads while the
    /// copy loop uploads the ones before them; mapped weights need no preparation.
    /// </summary>
    internal static Task MaterializeInBackground(List<BufferDecl> weights, CancellationToken ct)
    {
        const int Parallelism = 4;
        var lazy = weights.Select(b => b.Weight?.Source).OfType<IWeightSource>().Where(source => source.TryGetPointer() == 0).ToList();
        return Task.Run(() => Parallel.ForEach(lazy, new ParallelOptions { MaxDegreeOfParallelism = Parallelism, CancellationToken = ct }, source =>
        {
            try
            {
                source.Materialize();
            }
            catch (Exception)
            {
                // The producer's exception is cached and rethrown by the copy loop's read of this weight.
            }
        }));
    }

    /// <summary>
    /// Copies <paramref name="chunk"/> bytes of a weight (or constant) starting at <paramref name="done"/> into staging
    /// memory, repacking rows when the buffer asks for it. The source is a mapped pointer when it has one, otherwise
    /// the weight source or initial data is read through a span.
    /// </summary>
    internal static void CopyChunk(BufferDecl b, IWeightSource? src, nint ptr, long done, long chunk, byte* destination, long rowBytes)
    {
        var span = new Span<byte>(destination, (int)chunk);
        if (b.Repack != WeightRepack.None && ptr != 0)
        {
            RepackParallel(b, (byte*)ptr + done, destination, chunk, rowBytes);
        }
        else if (b.Repack != WeightRepack.None)
        {
            var tmp = ArrayPool<byte>.Shared.Rent((int)chunk);
            try
            {
                ReadChunk(b, src, done, tmp.AsSpan(0, (int)chunk));
                fixed (byte* tmpPointer = tmp)
                {
                    RepackParallel(b, tmpPointer, destination, chunk, rowBytes);
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(tmp);
            }
        }
        else if (b.InitialData is null && ptr != 0)
        {
            CopyParallel((byte*)ptr + done, destination, chunk);
        }
        else
        {
            ReadChunk(b, src, done, span);
        }
    }

    /// <summary>Reads a chunk of the buffer's initial data, or else of its weight source, through a span.</summary>
    private static void ReadChunk(BufferDecl b, IWeightSource? src, long done, Span<byte> destination)
    {
        if (b.InitialData is not null)
        {
            b.InitialData.AsSpan((int)done, destination.Length).CopyTo(destination);
        }
        else
        {
            src!.Read(done, destination);
        }
    }

    /// <summary>
    /// Multi-threaded copy: weight sources are usually mmapped files, where a single thread is bound by page-fault handling
    /// rather than memory bandwidth.
    /// </summary>
    private static void CopyParallel(byte* src, byte* dst, long bytes)
    {
        if (bytes <= CopyPieceBytes)
        {
            Buffer.MemoryCopy(src, dst, bytes, bytes);
            return;
        }

        nint source = (nint)src;
        nint destination = (nint)dst;
        Parallel.For(0, (bytes + CopyPieceBytes - 1) / CopyPieceBytes, piece =>
        {
            long offset = piece * CopyPieceBytes;
            long length = Math.Min(CopyPieceBytes, bytes - offset);
            Buffer.MemoryCopy((byte*)source + offset, (byte*)destination + offset, length, length);
        });
    }

    private static void RepackParallel(BufferDecl b, byte* src, byte* dst, long bytes, long rowBytes)
    {
        long rows = bytes / rowBytes;
        nint source = (nint)src;
        nint destination = (nint)dst;
        Parallel.For(0, (rows + RepackRowsPerTask - 1) / RepackRowsPerTask, task =>
        {
            long firstRow = task * RepackRowsPerTask;
            long taskBytes = Math.Min(RepackRowsPerTask, rows - firstRow) * rowBytes;
            WeightRepacker.Apply(b.Repack, b.RepackRowElements,
                new ReadOnlySpan<byte>((byte*)source + firstRow * rowBytes, (int)taskBytes),
                new Span<byte>((byte*)destination + firstRow * rowBytes, (int)taskBytes));
        });
    }
}
