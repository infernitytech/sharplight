using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Chat;
using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Serving.Tests;

public sealed class FakeLoaded : LoadedModel, IChatModel
{
    private readonly FakePool _pool;
    private readonly long _bytes;
    public FakeBatchDecoder Decoder { get; }
    public InferenceEngine Engine { get; }
    public ModelService Service { get; }
    public bool Disposed { get; private set; }
    /// <summary>Calls of <see cref="Dispose"/> (a host must make exactly one).</summary>
    public int DisposeCalls => _disposeCalls;
    /// <summary>Set when <see cref="Dispose"/> was entered.</summary>
    public ManualResetEventSlim DisposeStarted { get; } = new();
    /// <summary>When set, <see cref="Dispose"/> waits for it before it frees anything (a large model taking seconds to unload).</summary>
    public ManualResetEventSlim? DisposeGate { get; set; }
    private int _disposeCalls;
    public FakeLoaded(FakePool pool, long bytes, ModelKind kind, FakeBatchDecoder dec)
    {
        _pool = pool;
        _bytes = bytes;
        Kind = kind;
        Decoder = dec;
        pool.Charge(bytes);
        Engine = new InferenceEngine(dec, new FakeTokenizer());
        Service = new ModelService(Engine, ApiHost.Template, new ModelServiceOptions { ModelName = "x", DefaultMaxTokens = 64 });
    }
    public override ModelKind Kind { get; }
    public override long ResidentBytes => _bytes;
    public override bool IsIdle
    {
        get
        {
            var m = Engine.GetMetrics();
            return m.ActiveSequences == 0 && m.QueueDepth == 0;
        }
    }

    public override Task<bool> DrainAsync(TimeSpan timeout, CancellationToken cancellationToken = default) => Engine.DrainAsync(timeout, cancellationToken);
    public override void Dispose()
    {
        Interlocked.Increment(ref _disposeCalls);
        DisposeStarted.Set();
        DisposeGate?.Wait(TimeSpan.FromSeconds(30));
        if (Disposed)
        {
            return;
        }

        Disposed = true;
        Engine.Dispose();
        _pool.Charge(-_bytes);
    }
}
