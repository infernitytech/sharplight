using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Platform;
using Sharplight.Studio.ViewModels;

namespace Sharplight.Studio.Tests;

/// <summary>The native menus: the application name, the application menu and the menu bar items reach the right places.</summary>
public sealed class AppMenuTests
{
    private static NativeMenuItem Item(NativeMenu menu, string header) => menu.Items.OfType<NativeMenuItem>().Single(item => item.Header == header);

    private static Task RunAsync(NativeMenuItem item) => ((IAsyncRelayCommand)item.Command!).ExecuteAsync(item.CommandParameter);

    [Fact]
    public Task ApplicationName_IsTheProductName() => HeadlessApp.RunAsync(() =>
    {
        Assert.Equal("Sharplight Studio", Application.Current!.Name);
        return Task.CompletedTask;
    });

    [Fact]
    public async Task ApplicationMenu_HasAboutSettingsAndUpdates_ThatNavigate()
    {
        using var services = new TestServices();
        var viewModel = new MainWindowViewModel(services.Services);
        var menu = AppMenu.BuildApplicationMenu(viewModel);

        Assert.Equal(["About Sharplight Studio", "Settings…", "Check for Updates…"], menu.Items.OfType<NativeMenuItem>().Where(item => item is not NativeMenuItemSeparator).Select(item => item.Header));
        Assert.Contains(menu.Items, item => item is NativeMenuItemSeparator);
        Assert.Equal(new KeyGesture(Key.OemComma, KeyModifiers.Meta), Item(menu, "Settings…").Gesture);

        Item(menu, "About Sharplight Studio").Command!.Execute(null);
        Assert.True(viewModel.About.IsOpen);
        Assert.Equal(services.Updates.CurrentVersion, viewModel.About.Version);

        await RunAsync(Item(menu, "Settings…"));
        Assert.Same(viewModel.Settings, viewModel.CurrentPage);

        viewModel.CurrentPage = viewModel.Chat;
        await RunAsync(Item(menu, "Check for Updates…"));
        Assert.Same(viewModel.Settings, viewModel.CurrentPage);
        Assert.Equal(1, services.Updates.Checks);
    }

    [Fact]
    public async Task MenuBar_ViewNavigates_FileOpensLibrary_HelpOpensLinks()
    {
        using var services = new TestServices();
        var viewModel = new MainWindowViewModel(services.Services);
        var bar = AppMenu.BuildMenuBar(viewModel);
        Assert.Equal(["File", "View", "Help"], bar.Items.OfType<NativeMenuItem>().Select(item => item.Header));

        var view = Item(bar, "View").Menu!;
        Assert.Equal(viewModel.Pages.Select(page => page.Title), view.Items.OfType<NativeMenuItem>().Select(item => item.Header));
        var image = view.Items.OfType<NativeMenuItem>().Single(item => item.Header == "Image");
        await RunAsync(image);
        Assert.Same(viewModel.Image, viewModel.CurrentPage);

        Item(Item(bar, "File").Menu!, "Find Models on Hugging Face").Command!.Execute(null);
        Assert.Same(viewModel.Models, viewModel.CurrentPage);
        Assert.Equal(Sharplight.Studio.ViewModels.Models.ModelsPageViewModel.DiscoverTab, viewModel.Models.SelectedTab);

        var help = Item(bar, "Help").Menu!;
        var report = help.Items.OfType<NativeMenuItem>().Single(item => item.Header == "Report an Issue…");
        report.Command!.Execute(report.CommandParameter);
        Assert.Equal([AppMenu.IssuesUrl], services.Platform.Opened);
    }
}
