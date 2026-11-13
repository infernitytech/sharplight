namespace Sharplight.Vision.Tests;

/// <summary>Disposes a collection of resources together; the list may be filled after construction.</summary>
internal sealed class Disposer : IDisposable
{
    private readonly List<IDisposable> _items;

    public Disposer(out List<IDisposable> items)
    {
        _items = [];
        items = _items;
    }

    public Disposer(IEnumerable<IDisposable> items)
    {
        _items = [.. items];
    }

    public void Dispose()
    {
        foreach (var item in _items)
        {
            item.Dispose();
        }
    }
}
