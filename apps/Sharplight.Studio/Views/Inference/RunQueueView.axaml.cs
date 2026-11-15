using Avalonia.Controls;

namespace Sharplight.Studio.Views.Inference;

/// <summary>The run queue of an inference page: queued and running runs with progress and cancel.</summary>
internal sealed partial class RunQueueView : UserControl
{
    public RunQueueView()
    {
        InitializeComponent();
    }
}
