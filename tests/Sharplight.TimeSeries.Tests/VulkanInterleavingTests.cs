using Sharplight.Models;
using Sharplight.Runtime;

namespace Sharplight.TimeSeries.Tests;

/// <summary>
/// The encoder and the decoder of a Chronos model are two compiled models on one device: a decoder pass must not change what the encoder computes next
/// (regression test for #88, where it did on NVIDIA's Vulkan driver with chronos-t5-large).
/// </summary>
public class VulkanInterleavingTests
{
    [Theory]
    [InlineData("chronos-t5-small", false)]
    [InlineData("chronos-t5-small", true)]
    [InlineData("chronos-t5-large", false)]
    [InlineData("chronos-t5-large", true)]
    public void Decoder_pass_leaves_the_encoder_unchanged(string name, bool gpu)
    {
        TimeSeriesModels.Require(TimeSeriesModels.Model(name));
        using var model = TimeSeriesModels.Load(TimeSeriesModels.Open(gpu), TimeSeriesModels.Model(name));
        using var encoder = model.CompileComponent(ModelComponentRoles.Encoder);
        using var decoder = model.CompileComponent(ModelComponentRoles.Decoder);
        const int rows = 64;
        var before = Encode(encoder, rows);
        decoder.SetInput<int>(decoder.Graph.Input(InputRole.TokenIds), [0]);
        decoder.SetInput<float>(decoder.Graph.Input(InputRole.EncoderOutput), before);
        decoder.SetInput<int>(decoder.Graph.Input(InputRole.RowSpans), [0, rows]);
        decoder.SetInput<int>(decoder.Graph.Input(InputRole.OutputRows), [0]);
        var run = new RunParameters();
        run.Symbols[WellKnown.TokenSymbol] = 1;
        run.Symbols[WellKnown.EncoderSymbol] = rows;
        run.Symbols["R"] = 1;
        run.Scalars[WellKnown.PastScalar] = 0;
        run.Scalars[WellKnown.SourcePastScalar] = 0;
        decoder.Run(run);
        var logits = new float[(int)decoder.Graph.Outputs[WellKnown.LogitsOutput].Shape[-1].Fixed];
        decoder.GetOutput<float>(WellKnown.LogitsOutput, logits);
        Assert.All(logits, v => Assert.True(float.IsFinite(v)));
        Assert.Equal(before, Encode(encoder, rows));
    }

    private static float[] Encode(CompiledModel encoder, int rows)
    {
        encoder.SetInput<int>(encoder.Graph.Input(InputRole.TokenIds), [.. Enumerable.Range(0, rows).Select(i => 2000 + (i % 50))]);
        var run = new RunParameters();
        run.Symbols[WellKnown.TokenSymbol] = rows;
        encoder.Run(run);
        var output = new float[rows * (int)encoder.Graph.Outputs[WellKnown.EncoderOutput].Shape[-1].Fixed];
        encoder.GetOutput<float>(WellKnown.EncoderOutput, output);
        return output;
    }
}
