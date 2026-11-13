namespace Sharplight.Templating;

internal sealed class Token
{
    public TokenKind Kind;
    public string Text = "";
    public object? Value;
    public int Line;
    public int Col;

    public override string ToString() => Kind switch
    {
        TokenKind.Eof => "end of template",
        TokenKind.Data => "text",
        TokenKind.VarBegin => "'{{'",
        TokenKind.VarEnd => "'}}'",
        TokenKind.BlockBegin => "'{%'",
        TokenKind.BlockEnd => "'%}'",
        TokenKind.String => "string",
        _ => $"'{Text}'",
    };
}
