using System.Diagnostics;
using Sharplight.Models;

namespace Sharplight.Graph.Optimization;

/// <summary>
/// The device-independent graph optimization phase: runs an ordered pipeline of <see cref="IGraphPass"/>es over a graph (and,
/// first, over the branches of its control-flow nodes). After a round in which a pass with <see cref="IGraphPass.RepeatUntilStable"/>
/// changed the graph, the whole pipeline runs again, up to <see cref="MaxRounds"/> rounds.
/// </summary>
/// <remarks>
/// A model definition records the pipeline that optimized it in <see cref="OptimizedProperty"/>; loading a definition without the
/// current <see cref="PipelineId"/> optimizes it once (<see cref="PrepareForLoad"/>), a definition that carries it is used as is.
/// The phase can be switched off for debugging with the <see cref="DisableEnvironmentVariable"/> environment variable or the
/// loaders' options. Instances are immutable and thread-safe.
/// </remarks>
public sealed class GraphOptimizer
{
    /// <summary>
    /// Id of the default pipeline (<see cref="Default"/>). Changing what the default pipeline does changes this id, so stored
    /// definitions optimized by an older pipeline are optimized again when loaded.
    /// </summary>
    public const string PipelineId = "sharplight-ir/1";

    /// <summary>The <see cref="ModelDefinition.Properties"/> key that records the id of the pipeline that optimized the definition.</summary>
    public const string OptimizedProperty = "graph.optimized";

    /// <summary>Environment variable that disables the optimization at load time when set to <c>1</c> (for debugging and bisecting).</summary>
    public const string DisableEnvironmentVariable = "SHARPLIGHT_NO_GRAPH_OPT";

    /// <summary>Most pipeline rounds per graph.</summary>
    public const int MaxRounds = 8;

    /// <summary>The default pipeline (id <see cref="PipelineId"/>).</summary>
    public static GraphOptimizer Default { get; } = new(PipelineId, CreateDefaultPasses());

    /// <summary>
    /// Creates a pipeline.
    /// </summary>
    /// <param name="id">The id recorded in <see cref="OptimizedProperty"/> of definitions this pipeline optimizes.</param>
    /// <param name="passes">The passes, in the order they run in each round.</param>
    public GraphOptimizer(string id, IEnumerable<IGraphPass> passes)
    {
        Id = id;
        Passes = [.. passes];
    }

    /// <summary>The id recorded in definitions this pipeline optimizes.</summary>
    public string Id { get; }

    /// <summary>The passes, in the order they run in each round.</summary>
    public IReadOnlyList<IGraphPass> Passes { get; }

    /// <summary>
    /// The passes of the default pipeline: identity removal, view chain simplification, constant folding, common-subexpression
    /// elimination and dead-node elimination.
    /// </summary>
    public static IReadOnlyList<IGraphPass> CreateDefaultPasses() =>
        [new IdentityElimination(), new ViewChainSimplification(), new ConstantFolding(), new CommonSubexpressionElimination(), new DeadNodeElimination()];

    /// <summary>Whether <see cref="DisableEnvironmentVariable"/> is set to <c>1</c> (or <c>true</c>).</summary>
    public static bool DisabledByEnvironment => Environment.GetEnvironmentVariable(DisableEnvironmentVariable) is "1" or "true";

    /// <summary>Whether the definition was optimized by the pipeline with the given id.</summary>
    public static bool IsOptimized(ModelDefinition definition, string id = PipelineId)
        => definition.Properties.TryGetValue(OptimizedProperty, out var value) && value is string s && s == id;

    /// <summary>
    /// What loaders call before a definition's graphs are compiled: optimizes the definition with <see cref="Default"/> unless
    /// <paramref name="enabled"/> is false, <see cref="DisabledByEnvironment"/> is set, or it already carries <see cref="PipelineId"/>.
    /// </summary>
    /// <param name="definition">The loaded or freshly built definition.</param>
    /// <param name="enabled">The loader's option (false skips the phase).</param>
    /// <param name="log">Receives a one-line summary when the phase runs, or <see langword="null"/>.</param>
    /// <returns>The definition to use: <paramref name="definition"/> itself or its optimized copy.</returns>
    public static ModelDefinition PrepareForLoad(ModelDefinition definition, bool enabled = true, Action<string>? log = null)
    {
        if (!enabled || DisabledByEnvironment || IsOptimized(definition))
        {
            return definition;
        }

        var result = Default.Optimize(definition);
        log?.Invoke($"Graph optimization ({PipelineId}): {string.Join(", ", result.NodesBefore.Select(kv => $"{kv.Key} {kv.Value} -> {result.NodesAfter[kv.Key]} nodes"))} [{result.Elapsed.TotalMilliseconds:F0} ms]");
        return result.Definition;
    }

    /// <summary>
    /// Optimizes every graph of a definition. The result carries <see cref="OptimizedProperty"/> = <see cref="Id"/>; weights,
    /// states, metadata and owned resources are shared with <paramref name="definition"/>, which stays valid.
    /// </summary>
    /// <param name="definition">The definition to optimize (it is not modified).</param>
    public ModelOptimizationResult Optimize(ModelDefinition definition)
    {
        var watch = Stopwatch.StartNew();
        var statistics = new List<GraphPassStatistics>();
        var graphs = new Dictionary<string, ComputeGraph>(definition.Graphs.Count, StringComparer.Ordinal);
        var before = new Dictionary<string, int>(StringComparer.Ordinal);
        var after = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var (name, graph) in definition.Graphs)
        {
            var optimized = OptimizeGraph(graph, graph.Name, statistics);
            graphs[name] = optimized;
            before[name] = graph.Nodes.Count;
            after[name] = optimized.Nodes.Count;
        }

        var properties = new Dictionary<string, object>(definition.Properties) { [OptimizedProperty] = Id };
        var optimizedDefinition = new ModelDefinition
        {
            Architecture = definition.Architecture,
            Name = definition.Name,
            Weights = definition.Weights,
            Graphs = graphs,
            States = definition.States,
            Properties = properties,
            SourceMetadata = definition.SourceMetadata,
            OwnedResources = definition.OwnedResources,
        };
        return new ModelOptimizationResult(optimizedDefinition, statistics, before, after, watch.Elapsed);
    }

    /// <summary>Optimizes one graph (and the branches of its control-flow nodes).</summary>
    /// <param name="graph">The graph to optimize (it is not modified).</param>
    public GraphOptimizationResult Optimize(ComputeGraph graph)
    {
        var statistics = new List<GraphPassStatistics>();
        return new GraphOptimizationResult(OptimizeGraph(graph, graph.Name, statistics), statistics);
    }

    private ComputeGraph OptimizeGraph(ComputeGraph graph, string path, List<GraphPassStatistics> statistics)
    {
        graph = OptimizeSubgraphs(graph, path, statistics);
        for (int round = 1; round <= MaxRounds; round++)
        {
            bool repeat = false;
            foreach (var pass in Passes)
            {
                var watch = Stopwatch.StartNew();
                var next = pass.Run(graph);
                bool changed = !ReferenceEquals(next, graph);
                statistics.Add(new GraphPassStatistics(path, pass.Name, round, graph.Nodes.Count, next.Nodes.Count, changed, watch.Elapsed));
                if (changed)
                {
                    Validate(next, pass.Name);
                    repeat |= pass.RepeatUntilStable;
                }

                graph = next;
            }

            if (!repeat)
            {
                break;
            }
        }

        return graph;
    }

    /// <summary>Optimizes the subgraphs held in attributes of control-flow nodes (each is a self-contained graph).</summary>
    private ComputeGraph OptimizeSubgraphs(ComputeGraph graph, string path, List<GraphPassStatistics> statistics)
    {
        var rewriter = new GraphRewriter(graph);
        foreach (var n in graph.Nodes)
        {
            if (n.Op.Category != OpCategory.ControlFlow)
            {
                continue;
            }

            var attributes = n.Attributes;
            foreach (var (key, value) in n.Attributes)
            {
                if (value is ComputeGraph subgraph)
                {
                    var optimized = OptimizeGraph(subgraph, $"{path}/node#{n.Id}.{key}", statistics);
                    if (!ReferenceEquals(optimized, subgraph))
                    {
                        attributes = attributes.With(key, optimized);
                    }
                }
            }

            if (!ReferenceEquals(attributes, n.Attributes) && !rewriter.TryRewrite(n, n.Op, attributes, n.Inputs))
            {
                throw new InvalidOperationException($"Graph '{path}': optimizing the subgraphs of node {n.Id} ({n.Op.Name}) changed its output types.");
            }
        }

        return rewriter.Build();
    }

    /// <summary>Checks the invariants every pass must keep: operands are produced before use, and outputs are produced.</summary>
    private static void Validate(ComputeGraph graph, string pass)
    {
        var produced = new HashSet<Value>();
        foreach (var n in graph.Nodes)
        {
            foreach (var v in n.Inputs.Concat(n.Writes))
            {
                if (v.Kind == ValueKind.Intermediate && !produced.Contains(v))
                {
                    throw new InvalidOperationException($"Graph optimization pass '{pass}' left value {v} of graph '{graph.Name}' used before it is produced.");
                }
            }

            produced.UnionWith(n.Outputs);
        }

        foreach (var (name, v) in graph.Outputs)
        {
            if (v.Kind == ValueKind.Intermediate && !produced.Contains(v))
            {
                throw new InvalidOperationException($"Graph optimization pass '{pass}' dropped the producer of output '{name}' of graph '{graph.Name}'.");
            }
        }
    }
}
