using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.VideoGeneration;

namespace Sharplight.Architectures;

/// <summary>
/// Builds <see cref="VideoDiffusionGraphs.ImageEncode"/>: a vision transformer over the host-patchified image (the weights and the
/// embedding stage of <see cref="ImageTowerGraph"/>, class token, position table and pre-norm) that returns the hidden states of all rows
/// after <c>layers</c> of its layers, without the output norm: the penultimate hidden states of CLIP ViT-H/14 that condition Wan 2.1
/// image-to-video when built with one layer fewer than the tower has.
/// </summary>
internal static class VisionStatesGraphBuilder
{
    /// <summary>Builds the graph.</summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="options">The tower (its <see cref="TowerOptions.Layers"/> = the layers to run) and the patch and image sizes.</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, ImageTowerOptions options)
    {
        var b = new GraphBuilder(VideoDiffusionGraphs.ImageEncode, weights);
        int grid = options.ImageSize / options.Patch;
        var patches = b.Input(VideoDiffusionGraphs.Image, InputRole.Media, DataType.F32, new Shape(grid * grid, 3 * options.Patch * options.Patch));
        var x = ImageTowerGraph.Embed(b, patches, options, out int rows);
        x = TowerBlocks.Layers(b, x, options.Tower, rows, causal: false);
        b.Output(VideoDiffusionGraphs.ImageHidden, x);
        return b.Build();
    }

    /// <summary>Rows of the output: the class token and the patches.</summary>
    /// <param name="imageSize">The side of the square input.</param>
    /// <param name="patch">The patch side.</param>
    public static int Rows(int imageSize, int patch) => (imageSize / patch) * (imageSize / patch) + 1;
}
