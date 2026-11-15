namespace Sharplight.Hub;

/// <summary>One part of a multi-component repository, such as a Diffusers pipeline's <c>unet</c> or <c>vae</c>.</summary>
/// <param name="Name">The component's folder (<c>unet</c>), or the file name of a single-file model.</param>
/// <param name="Bytes">Size of the component's weights.</param>
/// <param name="Files">Number of weight files.</param>
public sealed record HubComponent(string Name, long Bytes, int Files);
