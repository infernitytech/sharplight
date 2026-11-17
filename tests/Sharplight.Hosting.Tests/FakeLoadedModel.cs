using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Hosting.Tests;

/// <summary>A loaded model that only holds memory of a <see cref="FakePool"/>.</summary>
public sealed class FakeLoadedModel : LoadedModel
{
    private readonly FakePool _pool;
    private readonly long _bytes;
    private bool _disposed;

    public FakeLoadedModel(FakePool pool, long bytes, ModelKind kind)
    {
        _pool = pool;
        _bytes = bytes;
        Kind = kind;
        pool.Charge(bytes);
    }

    public override ModelKind Kind { get; }

    public override long ResidentBytes => _bytes;

    public override void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _pool.Charge(-_bytes);
    }
}
