namespace Sharplight.Templating;

/// <summary>Python tuple: a list that prints with parentheses.</summary>
internal sealed class JinjaTuple : List<object?>
{
    public JinjaTuple() { }
    public JinjaTuple(IEnumerable<object?> items) : base(items) { }
}
