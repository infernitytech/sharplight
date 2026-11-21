using Sharplight.Cuda.Interop;

namespace Sharplight.Cuda;

/// <summary>
/// Side streams of a segment that let a captured step graph run independent launches side by side: <see cref="Fork"/> makes every
/// lane wait for the work queued so far on the main stream, launches then go to <see cref="Stream"/>, and <see cref="Join"/> makes the
/// main stream wait for the lanes used since the fork. Inside a stream capture these event records and waits become the edges of the
/// graph, so the launches between a fork and its join are parallel branches of one graph node set.
/// </summary>
internal sealed unsafe class CudaConcurrentLanes : IDisposable
{
    /// <summary>Side streams; the batch's first launch stays on the main stream.</summary>
    public const int LaneCount = 3;

    private readonly nint[] _streams = new nint[LaneCount];
    private readonly nint[] _done = new nint[LaneCount];
    private readonly bool[] _used = new bool[LaneCount];
    private nint _fork;

    /// <summary>Creates the side streams and their events (the caller has the device's context current).</summary>
    public CudaConcurrentLanes()
    {
        nint fork;
        CudaApi.Check(CudaApi.cuEventCreate(&fork, CudaApi.EventDisableTiming), "cuEventCreate");
        _fork = fork;
        for (int i = 0; i < LaneCount; i++)
        {
            nint stream;
            nint done;
            CudaApi.Check(CudaApi.cuStreamCreate(&stream, CudaApi.StreamNonBlocking), "cuStreamCreate");
            _streams[i] = stream;
            CudaApi.Check(CudaApi.cuEventCreate(&done, CudaApi.EventDisableTiming), "cuEventCreate");
            _done[i] = done;
        }
    }

    /// <summary>Marks the point of <paramref name="main"/> the lanes start from (each lane waits for it when it is first used).</summary>
    public void Fork(nint main)
    {
        CudaApi.Check(CudaApi.cuEventRecord(_fork, main), "cuEventRecord");
        Array.Clear(_used);
    }

    /// <summary>The side stream for the <paramref name="index"/>-th side launch since the fork (round robin over the lanes).</summary>
    public nint Stream(int index)
    {
        int lane = index % LaneCount;
        if (!_used[lane])
        {
            CudaApi.Check(CudaApi.cuStreamWaitEvent(_streams[lane], _fork, 0), "cuStreamWaitEvent");
            _used[lane] = true;
        }

        return _streams[lane];
    }

    /// <summary>Makes <paramref name="main"/> wait for every lane used since the fork.</summary>
    public void Join(nint main)
    {
        for (int lane = 0; lane < LaneCount; lane++)
        {
            if (_used[lane])
            {
                CudaApi.Check(CudaApi.cuEventRecord(_done[lane], _streams[lane]), "cuEventRecord");
                CudaApi.Check(CudaApi.cuStreamWaitEvent(main, _done[lane], 0), "cuStreamWaitEvent");
                _used[lane] = false;
            }
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        for (int i = 0; i < LaneCount; i++)
        {
            if (_done[i] != 0)
            {
                CudaApi.cuEventDestroy(_done[i]);
                _done[i] = 0;
            }

            if (_streams[i] != 0)
            {
                CudaApi.cuStreamDestroy(_streams[i]);
                _streams[i] = 0;
            }
        }

        if (_fork != 0)
        {
            CudaApi.cuEventDestroy(_fork);
            _fork = 0;
        }
    }
}
