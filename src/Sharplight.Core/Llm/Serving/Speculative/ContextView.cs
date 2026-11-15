using System.Collections;

namespace Sharplight.Llm.Serving.Speculative;

/// <summary>A read-only list of a slot's tokens followed by one more (the pending token), without copying.</summary>
internal sealed class ContextView : IReadOnlyList<int>
{
    private List<int> _tokens = [];
    private int _last;

    public void Set(List<int> tokens, int last)
    {
        _tokens = tokens;
        _last = last;
    }

    public int Count => _tokens.Count + 1;

    public int this[int index] => index == _tokens.Count ? _last : _tokens[index];

    public IEnumerator<int> GetEnumerator()
    {
        foreach (int t in _tokens)
        {
            yield return t;
        }

        yield return _last;
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
