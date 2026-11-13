namespace Sharplight.Templating;

/// <summary>Result of <c>|safe</c> / <c>|escape</c>: a string that is not escaped again, but escapes other operands when used with <c>+</c>.</summary>
internal sealed class Markup
{
    public Markup(string value) => Value = value;
    public string Value { get; }
    public override string ToString() => Value;

    internal static string Escape(string s)
    {
        var sb = new System.Text.StringBuilder(s.Length);
        foreach (char c in s)
        {
            switch (c)
            {
                case '&': sb.Append("&amp;"); break;
                case '<': sb.Append("&lt;"); break;
                case '>': sb.Append("&gt;"); break;
                case '\'': sb.Append("&#39;"); break;
                case '"': sb.Append("&#34;"); break;
                default: sb.Append(c); break;
            }
        }
        return sb.ToString();
    }
}
