using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Documents;
using Avalonia.Input.Platform;
using Avalonia.Layout;
using Avalonia.Media;
using Avalonia.Threading;
using Markdig;
using Markdig.Extensions.Tables;
using Markdig.Extensions.TaskLists;
using Markdig.Syntax;
using Markdig.Syntax.Inlines;
using MdInline = Markdig.Syntax.Inlines.Inline;

namespace Sharplight.Studio.Controls;

/// <summary>
/// Renders Markdown (CommonMark plus tables, task lists and strikethrough, parsed by Markdig) as selectable text blocks, with code
/// blocks that have a language label and a copy button. While text streams in, rebuilds are coalesced to a few per second. The
/// look comes from style classes (<c>md-h1</c>..<c>md-h4</c>, <c>md-p</c>, <c>md-code</c>, <c>md-quote</c>, <c>md-table</c>).
/// </summary>
internal sealed class MarkdownView : Decorator
{
    /// <summary>The Markdown text.</summary>
    public static readonly StyledProperty<string?> MarkdownProperty = AvaloniaProperty.Register<MarkdownView, string?>(nameof(Markdown));

    private static readonly MarkdownPipeline s_pipeline = new MarkdownPipelineBuilder()
        .UsePipeTables().UseTaskLists().UseEmphasisExtras().UseAutoLinks().Build();

    private static readonly TimeSpan s_coalesce = TimeSpan.FromMilliseconds(90);
    private static readonly FontFamily s_mono = new("Cascadia Code, JetBrains Mono, Consolas, DejaVu Sans Mono, Menlo, monospace");

    private readonly StackPanel _panel = new() { Spacing = 8 };
    private DispatcherTimer? _timer;
    private DateTime _lastBuild;

    static MarkdownView()
    {
        MarkdownProperty.Changed.AddClassHandler<MarkdownView>((view, _) => view.ScheduleBuild());
    }

    /// <summary>Creates the view.</summary>
    public MarkdownView()
    {
        Child = _panel;
    }

    /// <summary>The Markdown text.</summary>
    public string? Markdown
    {
        get => GetValue(MarkdownProperty);
        set => SetValue(MarkdownProperty, value);
    }

    /// <summary>The monospace font of code.</summary>
    public static FontFamily MonoFont => s_mono;

    private void ScheduleBuild()
    {
        var now = DateTime.UtcNow;
        if (now - _lastBuild >= s_coalesce)
        {
            Build();
            return;
        }

        if (_timer is null)
        {
            _timer = new DispatcherTimer { Interval = s_coalesce };
            _timer.Tick += (_, _) =>
            {
                _timer!.Stop();
                Build();
            };
        }

        if (!_timer.IsEnabled)
        {
            _timer.Start();
        }
    }

    private void Build()
    {
        _lastBuild = DateTime.UtcNow;
        _panel.Children.Clear();
        var text = Markdown;
        if (string.IsNullOrEmpty(text))
        {
            return;
        }

        var document = Markdig.Markdown.Parse(text, s_pipeline);
        foreach (var block in document)
        {
            if (CreateBlock(block) is { } control)
            {
                _panel.Children.Add(control);
            }
        }
    }

    private Control? CreateBlock(Block block)
    {
        switch (block)
        {
            case HeadingBlock heading:
                {
                    var title = TextBlockFor(heading.Inline);
                    title.Classes.Add("md-h" + Math.Clamp(heading.Level, 1, 4).ToString(System.Globalization.CultureInfo.InvariantCulture));
                    return title;
                }

            case ParagraphBlock paragraph:
                {
                    var body = TextBlockFor(paragraph.Inline);
                    body.Classes.Add("md-p");
                    return body;
                }

            case FencedCodeBlock fenced:
                return CodeBlock(fenced.Lines.ToString(), fenced.Info);

            case CodeBlock code:
                return CodeBlock(code.Lines.ToString(), null);

            case ListBlock list:
                return ListFor(list);

            case QuoteBlock quote:
                {
                    var inner = new StackPanel { Spacing = 6 };
                    foreach (var child in quote)
                    {
                        if (CreateBlock(child) is { } control)
                        {
                            inner.Children.Add(control);
                        }
                    }

                    var border = new Border { Child = inner };
                    border.Classes.Add("md-quote");
                    return border;
                }

            case ThematicBreakBlock:
                {
                    var rule = new Border { Height = 1 };
                    rule.Classes.Add("md-rule");
                    return rule;
                }

            case Table table:
                return TableFor(table);

            case HtmlBlock html:
                {
                    var raw = new SelectableTextBlock { Text = html.Lines.ToString(), TextWrapping = TextWrapping.Wrap, FontFamily = s_mono };
                    raw.Classes.Add("md-p");
                    return raw;
                }

            case ContainerBlock container:
                {
                    var inner = new StackPanel { Spacing = 6 };
                    foreach (var child in container)
                    {
                        if (CreateBlock(child) is { } control)
                        {
                            inner.Children.Add(control);
                        }
                    }

                    return inner;
                }

            default:
                return null;
        }
    }

    private Control ListFor(ListBlock list)
    {
        var panel = new StackPanel { Spacing = 4 };
        int number = int.TryParse(list.OrderedStart, out int start) ? start : 1;
        foreach (var item in list.OfType<ListItemBlock>())
        {
            var content = new StackPanel { Spacing = 4 };
            bool? task = null;
            foreach (var child in item)
            {
                if (child is ParagraphBlock { Inline: { } inline } && inline.FirstChild is TaskList taskList)
                {
                    task = taskList.Checked;
                }

                if (CreateBlock(child) is { } control)
                {
                    content.Children.Add(control);
                }
            }

            string marker = task switch
            {
                true => "☑",
                false => "☐",
                _ => list.IsOrdered ? $"{number}." : "•",
            };
            number++;
            var bullet = new TextBlock { Text = marker, MinWidth = list.IsOrdered ? 22 : 14, VerticalAlignment = VerticalAlignment.Top };
            bullet.Classes.Add("md-bullet");
            var row = new DockPanel();
            DockPanel.SetDock(bullet, Dock.Left);
            row.Children.Add(bullet);
            row.Children.Add(content);
            panel.Children.Add(row);
        }

        var indent = new Border { Child = panel, Padding = new Thickness(6, 0, 0, 0) };
        return indent;
    }

    private Control TableFor(Table table)
    {
        var grid = new Grid();
        int columns = table.OfType<TableRow>().Select(row => row.Count).DefaultIfEmpty(0).Max();
        for (int c = 0; c < columns; c++)
        {
            grid.ColumnDefinitions.Add(new ColumnDefinition(GridLength.Auto));
        }

        int r = 0;
        foreach (var row in table.OfType<TableRow>())
        {
            grid.RowDefinitions.Add(new RowDefinition(GridLength.Auto));
            int c = 0;
            foreach (var cell in row.OfType<TableCell>())
            {
                var content = new StackPanel();
                foreach (var child in cell)
                {
                    if (CreateBlock(child) is { } control)
                    {
                        content.Children.Add(control);
                    }
                }

                var border = new Border { Child = content };
                border.Classes.Add(row.IsHeader ? "md-th" : "md-td");
                Grid.SetRow(border, r);
                Grid.SetColumn(border, c);
                grid.Children.Add(border);
                c++;
            }

            r++;
        }

        var frame = new Border { Child = grid, HorizontalAlignment = HorizontalAlignment.Left };
        frame.Classes.Add("md-table");
        return new ScrollViewer { Content = frame, HorizontalScrollBarVisibility = Avalonia.Controls.Primitives.ScrollBarVisibility.Auto, VerticalScrollBarVisibility = Avalonia.Controls.Primitives.ScrollBarVisibility.Disabled };
    }

    private Control CodeBlock(string code, string? language)
    {
        code = code.TrimEnd('\n', '\r');
        var label = new TextBlock { Text = string.IsNullOrWhiteSpace(language) ? "code" : language.Trim(), VerticalAlignment = VerticalAlignment.Center };
        label.Classes.Add("md-code-lang");
        var copy = new Button { Content = "Copy", HorizontalAlignment = HorizontalAlignment.Right };
        copy.Classes.Add("md-copy");
        copy.Click += async (_, _) =>
        {
            if (TopLevel.GetTopLevel(this)?.Clipboard is { } clipboard)
            {
                await clipboard.SetTextAsync(code);
                copy.Content = "Copied";
                await Task.Delay(1500);
                copy.Content = "Copy";
            }
        };
        var header = new DockPanel { LastChildFill = false };
        header.Children.Add(label);
        DockPanel.SetDock(copy, Dock.Right);
        header.Children.Add(copy);
        header.Classes.Add("md-code-header");

        var body = new SelectableTextBlock { Text = code, FontFamily = s_mono, TextWrapping = TextWrapping.NoWrap };
        body.Classes.Add("md-code-text");
        var scroller = new ScrollViewer
        {
            Content = body,
            HorizontalScrollBarVisibility = Avalonia.Controls.Primitives.ScrollBarVisibility.Auto,
            VerticalScrollBarVisibility = Avalonia.Controls.Primitives.ScrollBarVisibility.Disabled,
        };
        var stack = new DockPanel();
        DockPanel.SetDock(header, Dock.Top);
        stack.Children.Add(header);
        stack.Children.Add(scroller);
        var border = new Border { Child = stack };
        border.Classes.Add("md-code");
        return border;
    }

    private static SelectableTextBlock TextBlockFor(ContainerInline? inline)
    {
        var block = new SelectableTextBlock { TextWrapping = TextWrapping.Wrap };
        if (inline is not null)
        {
            var inlines = block.Inlines ??= [];
            AddInlines(inlines, inline);
        }

        return block;
    }

    private static void AddInlines(InlineCollection target, ContainerInline container)
    {
        foreach (var inline in container)
        {
            if (Convert(inline) is { } converted)
            {
                target.Add(converted);
            }
        }
    }

    private static Avalonia.Controls.Documents.Inline? Convert(MdInline inline)
    {
        switch (inline)
        {
            case LiteralInline literal:
                return new Run(literal.Content.ToString());

            case CodeInline code:
                {
                    var run = new Run(" " + code.Content + " ") { FontFamily = s_mono };
                    run.Classes.Add("md-inline-code");
                    return run;
                }

            case EmphasisInline emphasis:
                {
                    Span span = emphasis.DelimiterChar == '~'
                        ? new Span { TextDecorations = TextDecorations.Strikethrough }
                        : emphasis.DelimiterCount >= 2 ? new Bold() : new Italic();
                    AddInlines(span.Inlines, emphasis);
                    return span;
                }

            case LinkInline { IsImage: true } image:
                return new Run("[image: " + (image.FirstChild?.ToString() ?? image.Url) + "]");

            case LinkInline link:
                {
                    var span = new Span { TextDecorations = TextDecorations.Underline };
                    span.Classes.Add("md-link");
                    AddInlines(span.Inlines, link);
                    if (span.Inlines.Count == 0)
                    {
                        span.Inlines.Add(new Run(link.Url));
                    }

                    return span;
                }

            case AutolinkInline autolink:
                {
                    var span = new Span { TextDecorations = TextDecorations.Underline };
                    span.Classes.Add("md-link");
                    span.Inlines.Add(new Run(autolink.Url));
                    return span;
                }

            case LineBreakInline lineBreak:
                return lineBreak.IsHard ? new LineBreak() : new Run(" ");

            case HtmlInline html:
                return html.Tag.StartsWith("<br", StringComparison.OrdinalIgnoreCase) ? new LineBreak() : new Run(html.Tag);

            case HtmlEntityInline entity:
                return new Run(entity.Transcoded.ToString());

            case TaskList:
                return null;

            case ContainerInline nested:
                {
                    var span = new Span();
                    AddInlines(span.Inlines, nested);
                    return span;
                }

            default:
                return new Run(inline.ToString());
        }
    }
}
