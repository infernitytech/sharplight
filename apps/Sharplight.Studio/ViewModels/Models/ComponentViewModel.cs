using Sharplight.Hub;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>One part of a multi-component repository in the details (a Diffusers folder, a whisper.cpp model file).</summary>
internal sealed class ComponentViewModel
{
    /// <param name="component">The part.</param>
    public ComponentViewModel(HubComponent component)
    {
        Name = component.Name;
        Size = ByteFormat.Format(component.Bytes);
        Files = component.Files == 1 ? "1 file" : $"{component.Files} files";
    }

    /// <summary>The folder or file name.</summary>
    public string Name { get; }

    /// <summary>Size of its weights.</summary>
    public string Size { get; }

    /// <summary>Number of weight files.</summary>
    public string Files { get; }
}
