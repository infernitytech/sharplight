using Avalonia.Controls;

namespace Sharplight.Studio.Views.Inference;

/// <summary>The parameter panel of an inference page: presets and one control per parameter of the task options.</summary>
internal sealed partial class ParameterPanelView : UserControl
{
    public ParameterPanelView()
    {
        InitializeComponent();
    }
}
