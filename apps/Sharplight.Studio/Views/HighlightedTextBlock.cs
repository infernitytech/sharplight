using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Documents;
using Avalonia.Media;
using Sharplight.Studio.Services.Search;

namespace Sharplight.Studio.Views;

/// <summary>A selectable text block that shows <see cref="Segments"/>, with the matching pieces highlighted.</summary>
internal sealed class HighlightedTextBlock : SelectableTextBlock
{
    /// <summary>The pieces to show.</summary>
    public static readonly StyledProperty<IReadOnlyList<TextSegment>?> SegmentsProperty =
        AvaloniaProperty.Register<HighlightedTextBlock, IReadOnlyList<TextSegment>?>(nameof(Segments));

    private static readonly IBrush s_highlight = new SolidColorBrush(Color.FromArgb(110, 255, 196, 0));

    static HighlightedTextBlock() => SegmentsProperty.Changed.AddClassHandler<HighlightedTextBlock>((block, _) => block.Rebuild());

    /// <summary>The pieces to show.</summary>
    public IReadOnlyList<TextSegment>? Segments
    {
        get => GetValue(SegmentsProperty);
        set => SetValue(SegmentsProperty, value);
    }

    /// <inheritdoc/>
    protected override Type StyleKeyOverride => typeof(SelectableTextBlock);

    private void Rebuild()
    {
        Inlines?.Clear();
        if (Segments is null)
        {
            return;
        }

        foreach (var segment in Segments)
        {
            var run = new Run(segment.Text);
            if (segment.IsMatch)
            {
                run.Background = s_highlight;
                run.FontWeight = FontWeight.SemiBold;
            }

            Inlines!.Add(run);
        }
    }
}
