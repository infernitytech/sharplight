using Sharplight.Import.Diffusers;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>Which Diffusers pipelines the importer says it can load, judged from the repository's pipeline class and components.</summary>
public sealed class PipelineSupportTests
{
    [Theory]
    [InlineData("StableDiffusionPipeline", "unet,vae", true)]
    [InlineData("StableDiffusionXLPipeline", "unet,vae,text_encoder_2", true)]
    [InlineData("FluxPipeline", "transformer,vae", true)]
    [InlineData("WanPipeline", "transformer,vae", true)]
    [InlineData("WanPipeline", "transformer,transformer_2,vae", true)]
    [InlineData("WanImageToVideoPipeline", "transformer,transformer_2,vae", false)]
    [InlineData("WanImageToVideoPipeline", "transformer,image_encoder,vae", false)]
    [InlineData("StableDiffusion3Pipeline", "transformer,vae", true)]
    [InlineData("ChromaPipeline", "transformer,vae", true)]
    [InlineData("ZImagePipeline", "transformer,vae", true)]
    [InlineData("QwenImagePipeline", "transformer,vae", true)]
    [InlineData("QwenImageEditPipeline", "transformer,vae", false)]
    [InlineData("QwenImage21Pipeline", "transformer,vae", true)]
    [InlineData("LTXPipeline", "transformer,vae,text_encoder", true)]
    [InlineData("HunyuanVideoPipeline", "transformer,vae", false)]
    public void ClassifiesPipelines(string pipelineClass, string components, bool supported)
    {
        var support = new DiffusersImporter().ClassifyPipeline(pipelineClass, components.Split(','));

        Assert.NotNull(support);
        Assert.Equal(supported, support.Supported);
        Assert.Equal(supported, support.Reason.Length == 0);
    }

    [Fact]
    public void UnknownKindOfIndex_IsNotJudged()
    {
        Assert.Null(new DiffusersImporter().ClassifyPipeline("SomeTransformersThing", []));
    }
}
