using System.Collections.Immutable;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Graph;

/// <summary>
/// Appends a <see cref="StandardOps.Sample"/> node to a decoder graph so token choice runs on the device that computes the logits and
/// only token ids (and optional log-probabilities) need to be downloaded; the logits stay a graph output for callers that sample
/// themselves (skip their download with <see cref="Runtime.RunParameters.SkipOutputs"/>).
/// </summary>
/// <remarks>
/// Semantics of a sampled (non-greedy) row with temperature τ, maximum logit m and p_i = exp((x_i - m) / τ):
/// <list type="number">
/// <item>Top-k: keep the tokens whose logit is at least the K-th largest (ties with it are kept). When top-p is active the candidate set
/// is additionally limited to the <see cref="MaxNucleusCandidates"/> best tokens (top-k 0 then means K = <see cref="MaxNucleusCandidates"/>).</item>
/// <item>Min-p: drop tokens with p_i below MinP (p of the best token is 1).</item>
/// <item>Top-p: order the remaining tokens by logit (then id); keep the shortest prefix whose mass, normalized over the top-k set,
/// reaches TopP.</item>
/// <item>Draw from the kept tokens proportionally to p_i (Gumbel-max with noise derived from seed, counter and token id).</item>
/// </list>
/// </remarks>
public static class SamplingHead
{
    /// <summary>Name of the i32 [R, <see cref="SamplingRow.Width"/>] parameters input (role <see cref="InputRole.SamplingParameters"/>).</summary>
    public const string ParametersInput = "sampling_params";
    /// <summary>Name of the i32 [R] output with the chosen token per row.</summary>
    public const string TokensOutput = "sampled_tokens";
    /// <summary>Name of the f32 [R] output with the chosen token's log-probability per row (rows that ask for it).</summary>
    public const string LogprobsOutput = "sampled_logprobs";
    /// <summary>Largest candidate set of a top-p row (see the remarks).</summary>
    public const int MaxNucleusCandidates = 1024;

    /// <summary>
    /// True when <paramref name="graph"/> has an f32 <paramref name="logitsOutput"/> output of shape [R, V] with a symbolic row axis and
    /// no sampling head yet.
    /// </summary>
    public static bool CanAppend(ComputeGraph graph, string logitsOutput = "logits")
        => graph.Outputs.TryGetValue(logitsOutput, out var logits)
           && logits.Type == DataType.F32 && logits.Rank == 2 && logits.Shape[0].IsSymbolic && !logits.Shape[1].IsSymbolic
           && !graph.Outputs.ContainsKey(TokensOutput)
           && graph.TryInput(InputRole.SamplingParameters) is null;

    /// <summary>
    /// Returns a copy of <paramref name="graph"/> with the sampling parameters input, a <see cref="StandardOps.Sample"/> node over the
    /// logits and the <see cref="TokensOutput"/> / <see cref="LogprobsOutput"/> outputs. The original graph is unchanged (the new graph
    /// shares its nodes).
    /// </summary>
    /// <exception cref="ArgumentException"><see cref="CanAppend"/> is false.</exception>
    public static ComputeGraph Append(ComputeGraph graph, string logitsOutput = "logits")
    {
        if (!CanAppend(graph, logitsOutput))
        {
            throw new ArgumentException($"Graph '{graph.Name}' has no [R, V] f32 '{logitsOutput}' output to sample from, or already samples.");
        }

        var logits = graph.Outputs[logitsOutput];
        int nextValue = graph.Nodes.SelectMany(n => n.Inputs.Concat(n.Outputs)).Concat(graph.Inputs).Max(v => v.Id) + 1;
        int nextNode = graph.Nodes.Count == 0 ? 0 : graph.Nodes.Max(n => n.Id) + 1;
        var parameters = new Value(nextValue++, ParametersInput, DataType.I32, new Shape(logits.Shape[0], Dim.Of(SamplingRow.Width)), ValueKind.Input)
        {
            Input = new InputDefinition { Name = ParametersInput, Role = InputRole.SamplingParameters },
        };
        var inputs = ImmutableArray.Create(logits, parameters);
        var types = StandardOps.Sample.Infer(inputs, Attributes.Empty);
        var node = new Node(nextNode, StandardOps.Sample, inputs, Attributes.Empty, logits.Producer?.Group ?? "");
        var tokens = new Value(nextValue++, TokensOutput, types[0].Type, types[0].Shape, ValueKind.Intermediate) { Producer = node };
        var logprobs = new Value(nextValue, LogprobsOutput, types[1].Type, types[1].Shape, ValueKind.Intermediate) { Producer = node };
        node.Outputs = [tokens, logprobs];
        var outputs = new Dictionary<string, Value>(graph.Outputs)
        {
            [TokensOutput] = tokens,
            [LogprobsOutput] = logprobs,
        };
        return new ComputeGraph(graph.Name, [.. graph.Nodes, node], [.. graph.Inputs, parameters], outputs, graph.RuntimeScalars);
    }
}
