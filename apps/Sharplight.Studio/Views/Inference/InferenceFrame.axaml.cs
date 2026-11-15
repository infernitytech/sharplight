using Avalonia;
using Avalonia.Controls;

namespace Sharplight.Studio.Views.Inference;

/// <summary>
/// The frame of every inference page: model picker, the page's own <see cref="Workspace"/>, run queue, and a side panel with the
/// parameters, history and outputs gallery. Its data context is the page's <c>InferencePageViewModel</c>.
/// </summary>
internal sealed partial class InferenceFrame : UserControl
{
    /// <summary>The page's input and result area.</summary>
    public static readonly StyledProperty<object?> WorkspaceProperty = AvaloniaProperty.Register<InferenceFrame, object?>(nameof(Workspace));

    public InferenceFrame()
    {
        InitializeComponent();
        WorkspaceProperty.Changed.AddClassHandler<InferenceFrame>((frame, args) => frame.WorkspacePresenter.Content = args.NewValue);
    }

    /// <summary>The page's input and result area.</summary>
    public object? Workspace
    {
        get => GetValue(WorkspaceProperty);
        set => SetValue(WorkspaceProperty, value);
    }
}
