using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Media.Imaging;
using Avalonia.VisualTree;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Platform;
using Sharplight.Studio.ViewModels;
using Sharplight.Studio.ViewModels.Settings;
using Sharplight.Studio.Views;

namespace Sharplight.Studio.Tests;

/// <summary>The settings page is split into tabs: every card is in exactly one tab, deep links open the right tab and the tab is remembered.</summary>
public sealed class SettingsTabsTests
{
    private static readonly string? s_screenshots = Environment.GetEnvironmentVariable("SHARPLIGHT_STUDIO_SCREENSHOTS");

    private static readonly string[] s_cards =
    [
        "Appearance", "Audio", "Storage", "Models", "Load defaults", "Residency", "New chat defaults", "Engine features", "Updates", "Data", "About",
    ];

    private static readonly Dictionary<string, SettingsTab> s_expected = new(StringComparer.Ordinal)
    {
        ["Appearance"] = SettingsTab.General,
        ["Audio"] = SettingsTab.General,
        ["New chat defaults"] = SettingsTab.General,
        ["Storage"] = SettingsTab.Storage,
        ["Models"] = SettingsTab.Models,
        ["Load defaults"] = SettingsTab.Engine,
        ["Residency"] = SettingsTab.Engine,
        ["Engine features"] = SettingsTab.Engine,
        ["Updates"] = SettingsTab.Updates,
        ["Data"] = SettingsTab.DataAndAbout,
        ["About"] = SettingsTab.DataAndAbout,
    };

    private static HashSet<string> Headings(Window window) =>
        window.GetVisualDescendants().OfType<TextBlock>().Where(block => block.Classes.Contains("h3") && block.IsEffectivelyVisible).Select(block => block.Text ?? "").ToHashSet(StringComparer.Ordinal);

    private static void Capture(Window window, string name)
    {
        if (s_screenshots is null)
        {
            return;
        }

        using var frame = window.CaptureRenderedFrame();
        Directory.CreateDirectory(s_screenshots);
        frame!.Save(Path.Combine(s_screenshots, name), new PngBitmapEncoderOptions());
    }

    [Fact]
    public Task EveryCard_IsInExactlyOneTab() => HeadlessApp.RunAsync(async () =>
    {
        using var services = new TestServices();
        var page = new SettingsPageViewModel(services.Services);
        var window = new Window { Width = 1100, Height = 900, Content = new SettingsPage { DataContext = page } };
        window.Show();
        try
        {
            var seen = new Dictionary<string, List<SettingsTab>>(StringComparer.Ordinal);
            foreach (var tab in Enum.GetValues<SettingsTab>())
            {
                page.ShowTab(tab);
                await HeadlessApp.RenderAsync(window);
                foreach (var heading in Headings(window))
                {
                    seen.TryAdd(heading, []);
                    seen[heading].Add(tab);
                }

                Capture(window, $"studio-settings-{tab.ToString().ToLowerInvariant()}.png");
            }

            foreach (var card in s_cards)
            {
                Assert.True(seen.TryGetValue(card, out var tabs), card);
                Assert.Equal([s_expected[card]], tabs);
            }
        }
        finally
        {
            window.Close();
        }
    });

    [Theory]
    [InlineData(1100, 900)]
    [InlineData(520, 800)]
    public Task ResetButton_AlignsWithTheCards(int width, int height) => HeadlessApp.RunAsync(async () =>
    {
        using var services = new TestServices();
        var page = new SettingsPageViewModel(services.Services);
        var window = new Window { Width = width, Height = height, Content = new SettingsPage { DataContext = page } };
        window.Show();
        try
        {
            foreach (var tab in new[] { SettingsTab.General, SettingsTab.Engine, SettingsTab.Updates })
            {
                page.ShowTab(tab);
                await HeadlessApp.RenderAsync(window);
                var reset = window.GetVisualDescendants().OfType<Button>().Single(button => button.Name == "ResetButton");
                var title = window.GetVisualDescendants().OfType<TextBlock>().First(block => block.Classes.Contains("h1"));
                var card = window.GetVisualDescendants().OfType<Border>().First(border => border.Classes.Contains("card") && border.IsEffectivelyVisible);
                var resetBounds = reset.TranslatePoint(new Avalonia.Point(reset.Bounds.Width, reset.Bounds.Height / 2), window)!.Value;
                var cardBounds = card.TranslatePoint(new Avalonia.Point(card.Bounds.Width, 0), window)!.Value;
                var cardLeft = card.TranslatePoint(default, window)!.Value;
                var titleLeft = title.TranslatePoint(default, window)!.Value;
                var titleCentre = title.TranslatePoint(new Avalonia.Point(0, title.Bounds.Height / 2), window)!.Value;
                Assert.Equal(cardBounds.X, resetBounds.X, 1.0);
                Assert.Equal(cardLeft.X, titleLeft.X, 1.0);
                Assert.InRange(resetBounds.Y, titleLeft.Y, titleCentre.Y + title.Bounds.Height * 2);
                Capture(window, $"studio-settings-{tab.ToString().ToLowerInvariant()}-{width}.png");
            }
        }
        finally
        {
            window.Close();
        }
    });

    [Fact]
    public Task DeepLinks_OpenTheRightTab() => HeadlessApp.RunAsync(async () =>
    {
        using var services = new TestServices();
        var viewModel = new MainWindowViewModel(services.Services);
        var menu = AppMenu.BuildApplicationMenu(viewModel);
        var updates = menu.Items.OfType<NativeMenuItem>().Single(item => item.Header == "Check for Updates…");

        await ((IAsyncRelayCommand)updates.Command!).ExecuteAsync(null);
        Assert.Same(viewModel.Settings, viewModel.CurrentPage);
        Assert.Equal(SettingsTab.Updates, viewModel.Settings.SelectedTab);

        viewModel.CurrentPage = viewModel.Chat;
        await viewModel.OpenStorageSettingsCommand.ExecuteAsync(null);
        Assert.Equal(SettingsTab.Storage, viewModel.Settings.SelectedTab);
    });

    [Fact]
    public void SelectedTab_IsRemembered()
    {
        using var services = new TestServices();
        new SettingsPageViewModel(services.Services).ShowTab(SettingsTab.Models);

        Assert.Equal(SettingsTab.Models, new SettingsPageViewModel(services.Services).SelectedTab);
    }
}
