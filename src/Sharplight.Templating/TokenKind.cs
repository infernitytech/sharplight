namespace Sharplight.Templating;

internal enum TokenKind
{
    Data,
    VarBegin,
    VarEnd,
    BlockBegin,
    BlockEnd,
    Name,
    String,
    Int,
    Float,
    Op,
    Eof,
}
