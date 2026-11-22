using System.Runtime.InteropServices;

namespace Sharplight.Runtime.Experts;

/// <summary>
/// Reads through one io_uring (Linux 5.6+, raw system calls, no liburing): the reads of a batch are queued as <c>IORING_OP_READ</c> entries,
/// up to the ring's depth in flight, on files opened with <c>O_DIRECT</c>; a reaper thread waits for completions, finishes the batches and
/// queues what did not fit. Created through <see cref="TryCreate"/>, which fails where the kernel or a seccomp profile refuses io_uring.
/// </summary>
internal sealed unsafe class IoUringExpertReader : IExpertReader
{
    private const long SetupSyscall = 425;
    private const long EnterSyscall = 426;
    private const uint EnterGetEvents = 1;
    private const byte OpNop = 0;
    private const byte OpRead = 22;
    private const long SqRingOffset = 0;
    private const long CqRingOffset = 0x8000000;
    private const long SqesOffset = 0x10000000;
    private const int ProtReadWrite = 3;
    private const int MapSharedPopulate = 0x01 | 0x8000;
    private const int Interrupted = 4;
    private const ulong WakeUserData = ulong.MaxValue;
    private const int SqeBytes = 64;
    private const int CqeBytes = 16;

    private readonly int _ring;
    private readonly nint _sqMap;
    private readonly nuint _sqMapBytes;
    private readonly nint _cqMap;
    private readonly nuint _cqMapBytes;
    private readonly nint _sqes;
    private readonly nuint _sqesBytes;
    private readonly uint* _sqTail;
    private readonly uint _sqMask;
    private readonly uint* _sqArray;
    private readonly uint* _cqHead;
    private readonly uint* _cqTail;
    private readonly uint _cqMask;
    private readonly byte* _cqes;
    private readonly int _depth;
    private readonly Lock _lock = new();
    private readonly Pending?[] _inFlight;
    private readonly Stack<int> _freeTags = new();
    private readonly Queue<Pending> _waiting = new();
    private readonly Thread _reaper;
    private volatile bool _disposed;

    /// <summary>A read in the ring (or waiting for room), with what is still to read after a short completion.</summary>
    private sealed class Pending(ExpertRead read, ExpertReadBatch batch)
    {
        public ExpertRead Read { get; } = read;
        public ExpertReadBatch Batch { get; } = batch;
        public long Done { get; set; }
    }

    private IoUringExpertReader(int ring, in Parameters p, int depth)
    {
        _ring = ring;
        _depth = depth;
        _sqMapBytes = p.SqOffsets.Array + p.SqEntries * sizeof(uint);
        _cqMapBytes = p.CqOffsets.Cqes + p.CqEntries * (uint)CqeBytes;
        _sqesBytes = p.SqEntries * (nuint)SqeBytes;
        _sqMap = Map(_sqMapBytes, SqRingOffset);
        _cqMap = Map(_cqMapBytes, CqRingOffset);
        _sqes = Map(_sqesBytes, SqesOffset);
        byte* sq = (byte*)_sqMap;
        byte* cq = (byte*)_cqMap;
        _sqTail = (uint*)(sq + p.SqOffsets.Tail);
        _sqMask = *(uint*)(sq + p.SqOffsets.RingMask);
        _sqArray = (uint*)(sq + p.SqOffsets.Array);
        _cqHead = (uint*)(cq + p.CqOffsets.Head);
        _cqTail = (uint*)(cq + p.CqOffsets.Tail);
        _cqMask = *(uint*)(cq + p.CqOffsets.RingMask);
        _cqes = cq + p.CqOffsets.Cqes;
        _inFlight = new Pending?[depth];
        for (int i = depth - 1; i >= 0; i--)
        {
            _freeTags.Push(i);
        }

        _reaper = new Thread(Reap) { IsBackground = true, Name = "Sharplight io_uring reaper" };
        _reaper.Start();
    }

    /// <inheritdoc/>
    public string Name => $"io_uring qd{_depth}";

    /// <inheritdoc/>
    public bool SupportsDeviceDestinations => false;

    /// <summary>Sets up a ring of <paramref name="depth"/> entries, or says why io_uring is unavailable.</summary>
    /// <param name="depth">Reads in flight at most (rounded up to a power of two by the kernel).</param>
    /// <param name="reader">The reader, or null.</param>
    /// <param name="reason">Why it failed, or null.</param>
    public static bool TryCreate(int depth, out IoUringExpertReader? reader, out string? reason)
    {
        reader = null;
        reason = null;
        if (!OperatingSystem.IsLinux())
        {
            reason = "not Linux";
            return false;
        }

        var p = default(Parameters);
        long fd = LinuxFileNative.Syscall(SetupSyscall, depth, (long)&p, 0, 0, 0, 0);
        if (fd < 0)
        {
            int errno = Marshal.GetLastPInvokeError();
            reason = errno switch
            {
                1 => "io_uring_setup: EPERM (blocked by a seccomp profile or kernel.io_uring_disabled)",
                38 => "io_uring_setup: ENOSYS (kernel without io_uring)",
                _ => $"io_uring_setup: errno {errno}",
            };
            return false;
        }

        try
        {
            reader = new IoUringExpertReader((int)fd, p, (int)Math.Min(p.SqEntries, (uint)depth));
            return true;
        }
        catch (Exception e)
        {
            LinuxFileNative.Close((int)fd);
            reason = $"io_uring ring mapping failed: {e.Message}";
            return false;
        }
    }

    /// <inheritdoc/>
    public IExpertFile Open(string path) => ExpertFileHandle.Open(path, direct: true);

    /// <inheritdoc/>
    public ExpertReadBatch Submit(ReadOnlySpan<ExpertRead> reads)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var batch = new ExpertReadBatch(reads.Length);
        lock (_lock)
        {
            foreach (var read in reads)
            {
                if (read.Destination.HostAddress == 0)
                {
                    batch.FailRead(new NotSupportedException("The io_uring expert reader reads into host memory only."));
                    continue;
                }

                _waiting.Enqueue(new Pending(read, batch));
            }

            SubmitWaiting();
        }

        return batch;
    }

    /// <summary>Moves waiting reads into free ring entries and hands them to the kernel. Holds <see cref="_lock"/>.</summary>
    private void SubmitWaiting()
    {
        uint tail = *_sqTail;
        int added = 0;
        while (_waiting.Count > 0 && _freeTags.Count > 0)
        {
            var pending = _waiting.Dequeue();
            int tag = _freeTags.Pop();
            _inFlight[tag] = pending;
            var file = (ExpertFileHandle)pending.Read.File;
            WriteEntry(tail, OpRead, file.Descriptor, pending.Read.Offset + pending.Done, pending.Read.Destination.HostAddress + (nint)pending.Done,
                (uint)(pending.Read.Length - pending.Done), (ulong)tag);
            tail++;
            added++;
        }

        if (added == 0)
        {
            return;
        }

        Volatile.Write(ref *_sqTail, tail);
        Enter((uint)added, 0, 0);
    }

    private void WriteEntry(uint tail, byte op, int fd, long offset, nint address, uint length, ulong userData)
    {
        uint index = tail & _sqMask;
        byte* sqe = (byte*)_sqes + index * SqeBytes;
        new Span<byte>(sqe, SqeBytes).Clear();
        sqe[0] = op;
        *(int*)(sqe + 4) = fd;
        *(long*)(sqe + 8) = offset;
        *(long*)(sqe + 16) = address;
        *(uint*)(sqe + 24) = length;
        *(ulong*)(sqe + 32) = userData;
        _sqArray[index] = index;
    }

    /// <summary><c>io_uring_enter</c>, retried on EINTR.</summary>
    private long Enter(uint submit, uint minComplete, uint flags)
    {
        while (true)
        {
            long r = LinuxFileNative.Syscall(EnterSyscall, _ring, submit, minComplete, flags, 0, 0);
            if (r >= 0)
            {
                return r;
            }

            int errno = Marshal.GetLastPInvokeError();
            if (errno != Interrupted)
            {
                throw new IOException($"io_uring_enter failed: errno {errno}.");
            }
        }
    }

    private void Reap()
    {
        while (!_disposed)
        {
            try
            {
                Enter(0, 1, EnterGetEvents);
            }
            catch (IOException) when (_disposed)
            {
                return;
            }

            uint head = *_cqHead;
            uint tail = Volatile.Read(ref *_cqTail);
            var finished = new List<(Pending Read, int Result)>();
            for (; head != tail; head++)
            {
                byte* cqe = _cqes + (head & _cqMask) * CqeBytes;
                ulong userData = *(ulong*)cqe;
                int result = *(int*)(cqe + 8);
                if (userData == WakeUserData)
                {
                    continue;
                }

                lock (_lock)
                {
                    var pending = _inFlight[(int)userData]!;
                    _inFlight[(int)userData] = null;
                    _freeTags.Push((int)userData);
                    finished.Add((pending, result));
                }
            }

            Volatile.Write(ref *_cqHead, head);
            lock (_lock)
            {
                foreach (var (pending, result) in finished)
                {
                    Finish(pending, result);
                }

                if (!_disposed)
                {
                    SubmitWaiting();
                }
            }
        }
    }

    /// <summary>Completes a read, or queues its remainder after a short read that did not reach the end of the file. Holds <see cref="_lock"/>.</summary>
    private void Finish(Pending pending, int result)
    {
        if (result < 0)
        {
            pending.Batch.FailRead(new IOException($"io_uring read of '{pending.Read.File.Path}' at {pending.Read.Offset} failed: errno {-result}."));
            return;
        }

        pending.Done += result;
        bool atEnd = result == 0 || pending.Read.Offset + pending.Done >= pending.Read.File.Length;
        if (pending.Done < pending.Read.Length && !atEnd)
        {
            _waiting.Enqueue(pending);
            return;
        }

        pending.Batch.CompleteRead(pending.Done);
    }

    private nint Map(nuint bytes, long offset)
    {
        nint address = LinuxFileNative.Mmap(0, bytes, ProtReadWrite, MapSharedPopulate, _ring, offset);
        if (address == -1)
        {
            throw new IOException($"mmap of the io_uring ring failed: errno {Marshal.GetLastPInvokeError()}.");
        }

        return address;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        lock (_lock)
        {
            _disposed = true;
            // A no-op entry wakes the reaper out of its wait.
            uint tail = *_sqTail;
            WriteEntry(tail, OpNop, -1, 0, 0, 0, WakeUserData);
            Volatile.Write(ref *_sqTail, tail + 1);
            Enter(1, 0, 0);
        }

        _reaper.Join();
        LinuxFileNative.Munmap(_sqes, _sqesBytes);
        LinuxFileNative.Munmap(_cqMap, _cqMapBytes);
        LinuxFileNative.Munmap(_sqMap, _sqMapBytes);
        LinuxFileNative.Close(_ring);
    }

    /// <summary><c>struct io_uring_params</c>.</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct Parameters
    {
        public uint SqEntries;
        public uint CqEntries;
        public uint Flags;
        public uint SqThreadCpu;
        public uint SqThreadIdle;
        public uint Features;
        public uint WqFd;
        public uint Reserved0;
        public uint Reserved1;
        public uint Reserved2;
        public SqRingOffsets SqOffsets;
        public CqRingOffsets CqOffsets;
    }

    /// <summary><c>struct io_sqring_offsets</c>.</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct SqRingOffsets
    {
        public uint Head;
        public uint Tail;
        public uint RingMask;
        public uint RingEntries;
        public uint Flags;
        public uint Dropped;
        public uint Array;
        public uint Reserved;
        public ulong UserAddress;
    }

    /// <summary><c>struct io_cqring_offsets</c>.</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct CqRingOffsets
    {
        public uint Head;
        public uint Tail;
        public uint RingMask;
        public uint RingEntries;
        public uint Overflow;
        public uint Cqes;
        public uint Flags;
        public uint Reserved;
        public ulong UserAddress;
    }
}
