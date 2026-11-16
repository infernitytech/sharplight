namespace Sharplight.Vision.Sample;

/// <summary>The parsed command line shared by the sub-commands of <c>sharplight-vision</c>.</summary>
/// <param name="ModelPath">The model: an <c>.slm</c> package or a Hugging Face directory.</param>
/// <param name="ImagePath">The input image (PNG).</param>
/// <param name="OutputPath">The annotated PNG to write, or <see langword="null"/> for none.</param>
/// <param name="UseCpu">Run on the CPU instead of a GPU.</param>
/// <param name="DeviceIndex">The device index within the provider.</param>
/// <param name="Threshold">Detection score threshold.</param>
/// <param name="TopK">Classes to print.</param>
/// <param name="Labels">Zero-shot candidate prompts (an embedding model with a text tower classifies against them); empty for a plain classifier.</param>
/// <param name="Verbose">Print load and compile messages to standard error.</param>
/// <param name="Points">Segmentation points on the object, as <c>x,y</c> pixels of the image.</param>
/// <param name="NegativePoints">Segmentation points beside the object, as <c>x,y</c>.</param>
/// <param name="Box">The segmentation box as <c>x1,y1,x2,y2</c>, or <see langword="null"/>.</param>
/// <param name="TileSize">Upscaling: the tile side in input pixels (0: the whole image at once).</param>
/// <param name="TilePadding">Upscaling: the context shared by neighbouring tiles, in pixels.</param>
public sealed record VisionSettings(string ModelPath, string ImagePath, string? OutputPath, bool UseCpu, int DeviceIndex, float Threshold, int TopK,
    IReadOnlyList<string> Labels, bool Verbose, IReadOnlyList<string> Points, IReadOnlyList<string> NegativePoints, string? Box, int TileSize, int TilePadding);
