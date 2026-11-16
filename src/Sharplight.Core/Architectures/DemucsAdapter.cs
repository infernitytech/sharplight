using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// The hybrid transformer Demucs music source separator (canonical arch <c>demucs</c>, kind <see cref="ModelKinds.AudioSeparation"/>): builds
/// <see cref="SeparationGraphs.SeparateGraph"/> for the model's fixed segment (see <see cref="DemucsGraph"/>).
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>demucs.*</c>): <c>sources</c> (comma separated), <c>sample_rate</c>, <c>audio_channels</c>, <c>channels</c>, <c>growth</c>, <c>depth</c>, <c>nfft</c>, <c>kernel_size</c>,
/// <c>stride</c>, <c>dconv_depth</c>, <c>dconv_compress</c>, <c>bottom_channels</c>, <c>freq_emb</c>, <c>emb_scale</c>, <c>transformer.{layer_count, head_count, hidden_size, max_period,
/// position_weight}</c>, <c>segment_samples</c>. Weights keep the names of the reference implementation, with every convolution weight as a 4-D tensor (<c>[O, C, KH, KW]</c>, the
/// waveform branch with <c>KH</c> = 1) and the fused attention projections split into <c>{q,k,v}.{weight,bias}</c> next to <c>out_proj</c>.
/// </remarks>
public sealed class DemucsAdapter : IModelAdapter
{
    /// <inheritdoc/>
    public string Name => "Demucs source separator";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [DemucsParameters.Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        var p = DemucsParameters.From(src);
        var description = new ModelDescription
        {
            Kind = ModelKinds.AudioSeparation,
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.Encoder,
                    Graphs = [SeparationGraphs.SeparateGraph],
                    WeightPrefixes = [""],
                    Config = ComponentConfig.Of((SeparationGraphs.SampleRateKey, p.SampleRate), (SeparationGraphs.ChannelsKey, p.AudioChannels),
                        (SeparationGraphs.StemsKey, string.Join(',', p.Sources)), (SeparationGraphs.SegmentSamplesKey, p.SegmentSamples),
                        (SeparationGraphs.FftSizeKey, p.Nfft), (SeparationGraphs.HopKey, p.Hop)),
                },
            ],
        };
        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = new Dictionary<string, ComputeGraph>(StringComparer.Ordinal) { [SeparationGraphs.SeparateGraph] = DemucsGraph.Build(p, src.Weight) },
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };
    }
}
