using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Serving.Tests;

public sealed class FakeLoader : IModelLoader
{
    private readonly FakePool _pool;
    private readonly long _bytes;
    private readonly bool _offload;
    public int Prepares, Loads;
    public long? LastLimit;
    public Func<FakeBatchDecoder>? MakeDecoder;
    public TimeSpan LoadDelay;
    public string Reply = "ok";
    public List<FakeLoaded> Instances { get; } = [];
    public FakeLoader(FakePool pool, long bytes, bool offload = false)
    {
        _pool = pool;
        _bytes = bytes;
        _offload = offload;
    }
    public ModelKind Kind { get; init; } = ModelKind.Chat;

    public IPreparedModel Prepare(CancellationToken ct)
    {
        Interlocked.Increment(ref Prepares);
        return new Prepared(this);
    }

    private sealed class Prepared(FakeLoader o) : IPreparedModel
    {
        public IReadOnlyList<MemoryRequirement> Requirements => [new MemoryRequirement(o._pool, o._bytes)];
        public bool CanOffload => o._offload;
        public LoadedModel Load(long? limit)
        {
            Interlocked.Increment(ref o.Loads);
            o.LastLimit = limit;
            if (o.LoadDelay > TimeSpan.Zero)
            {
                Thread.Sleep(o.LoadDelay);
            }

            long bytes = limit is { }
            l ? Math.Min(l, o._bytes) : o._bytes;
            var dec = o.MakeDecoder?.Invoke() ?? new FakeBatchDecoder(2);
            dec.Script(o.Reply);
            var inst = new FakeLoaded(o._pool, bytes, o.Kind, dec);
            lock (o.Instances)
            {
                o.Instances.Add(inst);
            }

            return inst;
        }
        public void Dispose() { }
    }
}
