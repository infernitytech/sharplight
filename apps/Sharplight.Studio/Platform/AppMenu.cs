using Avalonia.Controls;
using Avalonia.Input;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.ViewModels;

namespace Sharplight.Studio.Platform;

/// <summary>
/// The native menus (macOS menu bar; Windows and Linux draw no native menu, there the same commands are the navigation, the Settings
/// page and the About dialog). <see cref="BuildApplicationMenu"/> fills the application menu (the one named after the app) with About,
/// Settings and Check for Updates, next to the standard Services, Hide and Quit items macOS adds itself; <see cref="BuildMenuBar"/>
/// builds File, View and Help.
/// </summary>
internal static class AppMenu
{
    /// <summary>The documentation.</summary>
    public const string DocumentationUrl = "https://github.com/infernitytech/sharplight/blob/main/docs/studio.md";

    /// <summary>The issue tracker.</summary>
    public const string IssuesUrl = "https://github.com/infernitytech/sharplight/issues/new";

    /// <summary>The items of the application menu.</summary>
    public static NativeMenu BuildApplicationMenu(MainWindowViewModel viewModel) =>
    [
        new NativeMenuItem($"About {AboutViewModel.ApplicationName}") { Command = viewModel.ShowAboutCommand },
        new NativeMenuItemSeparator(),
        new NativeMenuItem("Settings…") { Command = viewModel.OpenSettingsCommand, Gesture = new KeyGesture(Key.OemComma, KeyModifiers.Meta) },
        new NativeMenuItem("Check for Updates…") { Command = viewModel.CheckForUpdatesCommand },
    ];

    /// <summary>File, View and Help menus.</summary>
    public static NativeMenu BuildMenuBar(MainWindowViewModel viewModel)
    {
        var view = new NativeMenu();
        int number = 1;
        foreach (var page in viewModel.Pages)
        {
            view.Add(new NativeMenuItem(page.Title)
            {
                Command = viewModel.NavigateCommand,
                CommandParameter = page,
                Gesture = number <= 9 ? new KeyGesture(Key.D0 + number, KeyModifiers.Meta) : null,
            });
            number++;
        }

        return
        [
            new NativeMenuItem("File")
            {
                Menu =
                [
                    new NativeMenuItem("Model Library") { Command = new RelayCommand(viewModel.OpenLibrary) },
                    new NativeMenuItem("Find Models on Hugging Face") { Command = new RelayCommand(viewModel.OpenDiscover) },
                    new NativeMenuItem("Downloads") { Command = viewModel.OpenDownloadsCommand },
                ],
            },
            new NativeMenuItem("View") { Menu = view },
            new NativeMenuItem("Help")
            {
                Menu =
                [
                    new NativeMenuItem($"{AboutViewModel.ApplicationName} Documentation") { Command = viewModel.OpenLinkCommand, CommandParameter = DocumentationUrl },
                    new NativeMenuItem("Report an Issue…") { Command = viewModel.OpenLinkCommand, CommandParameter = IssuesUrl },
                ],
            },
        ];
    }
}
