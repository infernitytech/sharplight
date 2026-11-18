using Avalonia.Controls;
using Avalonia.Media;

namespace Sharplight.Studio.Tests;

/// <summary>The checker's extreme-wrap rule: wrapping text squeezed to a few characters of width is a layout bug.</summary>
public sealed class LayoutCheckerTests
{
    [Fact]
    public Task TextSqueezedToAFewCharacters_IsReported_ButNormalWrappingAndShortWordsAreNot() => HeadlessApp.RunAsync(async () =>
    {
        const string Sentence = "Optional: start from an image (image to image). Drop a file here or choose one.";
        var squeezed = new TextBlock { Text = Sentence, TextWrapping = TextWrapping.Wrap, Width = 40 };
        var normal = new TextBlock { Text = Sentence, TextWrapping = TextWrapping.Wrap, Width = 200 };
        var narrowShortWord = new TextBlock { Text = "OK", TextWrapping = TextWrapping.Wrap, Width = 12 };
        var noWrap = new TextBlock { Text = Sentence, TextWrapping = TextWrapping.NoWrap, Width = 40 };
        var panel = new StackPanel { Children = { squeezed, normal, narrowShortWord, noWrap } };
        var window = new Window { Content = panel, Width = 400, Height = 800 };
        window.Show();
        try
        {
            await HeadlessApp.RenderAsync(window);

            Assert.NotNull(LayoutChecker.FindExtremeWrap(squeezed));
            Assert.Null(LayoutChecker.FindExtremeWrap(normal));
            Assert.Null(LayoutChecker.FindExtremeWrap(narrowShortWord));
            Assert.Null(LayoutChecker.FindExtremeWrap(noWrap));
            var problems = LayoutChecker.Find(window);
            Assert.Single(problems, problem => problem.Contains("wraps in", StringComparison.Ordinal));
        }
        finally
        {
            window.Close();
        }
    });
}
