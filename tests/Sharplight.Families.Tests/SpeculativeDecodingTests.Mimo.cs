using Sharplight.Llm.Serving;
using Sharplight.Models;

namespace Sharplight.Families.Tests;

public partial class SpeculativeDecodingTests
{
    private static readonly string MimoModelPath = Path.Combine(Path.GetDirectoryName(ModelPath)!, "tiny-random-mimo-mtp");

    /// <summary>
    /// MiMo (HF <c>mimo</c>: Qwen2 layers plus <c>model.mtp_layers.0</c>, whose input projection reads [hidden | embedding]) imports its MTP
    /// layer as the NextN draft head: one draft layer, and greedy output with it is token-identical.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task MimoMtpHead_GreedyOutput_Matches_NonSpeculativeDecoding(bool vulkan)
        => await Matches_NonSpeculativeDecoding(MimoModelPath, vulkan, "mtp", vulkan, SamplingParams.GreedyDefault,
            m => Assert.Equal(1, m.Definition.Properties[WellKnown.DraftLayers]));

    /// <summary>
    /// The MiMo head proposes the reference drafts: transformers' Qwen2 trunk (final-norm hidden states) and its Qwen2DecoderLayer loaded
    /// with the MTP layer plus MiMo's norms and input projection (<c>tools/mk_tiny_mimo_mtp.py</c>).
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task MimoMtpHead_Proposes_ReferenceDrafts(bool vulkan)
        => await Proposes_ReferenceDrafts(MimoModelPath, [], Path.Combine(MimoModelPath, "mimo_mtp_reference.json"), vulkan, cpuFallback: false);
}
