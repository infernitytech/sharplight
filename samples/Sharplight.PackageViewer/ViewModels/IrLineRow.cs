using Sharplight.Package;

namespace Sharplight.PackageViewer.ViewModels;

/// <summary>One line of the IR view.</summary>
/// <param name="Index">Line index (0-based).</param>
/// <param name="Text">The line with its indentation.</param>
/// <param name="Line">The line's structure (scope, node, defined and used values).</param>
public sealed record IrLineRow(int Index, string Text, IrLine Line)
{
    private const int PreviewLength = 140;

    /// <summary>1-based line number for display.</summary>
    public int Number => Index + 1;

    /// <summary>A shortened copy of the line for reference lists.</summary>
    public string Preview
    {
        get
        {
            var text = Text.TrimStart();
            return text.Length <= PreviewLength ? text : text[..PreviewLength] + "...";
        }
    }
}
