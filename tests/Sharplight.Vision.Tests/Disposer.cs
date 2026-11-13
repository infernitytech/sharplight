namespace Sharplight.Vision.Tests;

internal sealed class Disposer : IDisposable
{
    private readonly List<IDisposable> _items;
    public Disposer(out List<IDisposable> items) { items = _items = []; }
    public Disposer(IEnumerable<IDisposable> items) { _items = [.. items]; }
    public void Dispose() { foreach (var i in _items)
        {
            i.Dispose();
        }
    }
}
