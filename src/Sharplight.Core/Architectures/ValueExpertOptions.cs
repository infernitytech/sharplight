using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Routing of the value experts of <see cref="ValueExperts"/> (MoVA), read from the <c>{arch}.attention.value_experts.*</c> keys:
/// <c>count</c> (E, 0 = none), <c>used_count</c> (k), <c>router_activation</c> (as <c>moe.router_activation</c>, default "softmax"),
/// <c>normalize_weights</c> (default: true when k &gt; 1) and <c>weights_scale</c> (default 1). The selection bias, when present, only decides
/// which experts are picked (see <see cref="MoeRouter"/>).
/// </summary>
/// <param name="Routing">The routing in the terms of the expert router.</param>
internal sealed record ValueExpertOptions(MoeOptions Routing)
{
    private const string Prefix = "attention.value_experts.";

    /// <summary>The options of <paramref name="source"/>, or null when the model has no value experts.</summary>
    /// <exception cref="InvalidDataException">The counts are inconsistent or the router activation is unknown.</exception>
    public static ValueExpertOptions? Read(ModelSource source)
    {
        int experts = source.Arch(Prefix + "count", 0);
        if (experts == 0)
        {
            return null;
        }

        int used = source.Arch(Prefix + "used_count", 0);
        var routing = new MoeOptions(
            Experts: experts,
            ExpertsUsed: used,
            ScoreFunction: MoeOptions.ScoreFunctionOf(source, Prefix + "router_activation"),
            NormalizeWeights: source.Arch(Prefix + "normalize_weights", used > 1),
            WeightsScale: source.Arch(Prefix + "weights_scale", 1f));
        if (!routing.IsValid)
        {
            throw new InvalidDataException($"{source.Architecture}: {Prefix}used_count must be between 1 and {Prefix}count.");
        }

        return new ValueExpertOptions(routing);
    }
}
