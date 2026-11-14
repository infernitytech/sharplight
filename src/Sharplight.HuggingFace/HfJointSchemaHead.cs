using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// The joint schema head of Cloudflare Clef checkpoints: <c>joint_head_config.json</c> (hidden_size, width, routing_layers, layers, heads,
/// feedforward) and <c>joint_head.safetensors</c> (a <c>JointSchemaHead</c> of the repo's joint_schema_model.py) beside a Qwen3.5 checkpoint,
/// imported as the canonical "joint_schema" span scoring head. <c>nn.MultiheadAttention</c>'s packed <c>in_proj</c> is split into q, k, v and
/// the three learned scalars become <c>span_head.scales</c> = (exp(min(prior_logit_scale, ln 100)), exp(min(joint_logit_scale, ln 100)),
/// sigmoid(residual_gate)).
/// </summary>
internal static partial class HfJointSchemaHead
{
    private const string ConfigFile = "joint_head_config.json";
    private const string WeightsFile = "joint_head.safetensors";
    private const string P = "span_head.";

    private static readonly Dictionary<string, string> s_global = new(StringComparer.Ordinal)
    {
        ["hidden_norm"] = "hidden_norm",
        ["memory_projection"] = "memory",
        ["question_projection"] = "question",
        ["option_question_projection"] = "option_question",
        ["global_projection"] = "global",
        ["option_context_projection"] = "option_context",
        ["option_lexical_projection"] = "option_lexical",
        ["type_embedding"] = "type_embedding",
        ["option_summary_norm"] = "option_summary_norm",
        ["field_norm"] = "field_norm",
        ["option_norm"] = "option_norm",
        ["residual_scorer.0"] = "scorer",
        ["residual_scorer.3"] = "scorer_out",
    };

    private static readonly Dictionary<string, string> s_routing = new(StringComparer.Ordinal)
    {
        ["query_norm"] = "query_norm",
        ["memory_norm"] = "memory_norm",
        ["attention.in_proj"] = "attention",
        ["attention.out_proj"] = "attention.o",
        ["feedforward_norm"] = "ffn_norm",
        ["feedforward.0"] = "ffn_up",
        ["feedforward.3"] = "ffn_down",
    };

    private static readonly Dictionary<string, string> s_joint = new(StringComparer.Ordinal)
    {
        ["norm1"] = "self_norm",
        ["self_attn.in_proj"] = "self_attention",
        ["self_attn.out_proj"] = "self_attention.o",
        ["norm2"] = "cross_norm",
        ["multihead_attn.in_proj"] = "cross_attention",
        ["multihead_attn.out_proj"] = "cross_attention.o",
        ["norm3"] = "ffn_norm",
        ["linear1"] = "ffn_up",
        ["linear2"] = "ffn_down",
    };

    private static readonly string[] s_scalars = ["prior_logit_scale", "joint_logit_scale", "residual_gate"];

    [GeneratedRegex(@"^(evidence_layers|layers)\.(\d+)\.(.+?)(?:\.|_)(weight|bias)$")]
    private static partial Regex LayerTensor();

    [GeneratedRegex(@"^(.+)\.(weight|bias)$")]
    private static partial Regex GlobalTensor();

    /// <summary>Whether <paramref name="dir"/> holds a joint schema head.</summary>
    public static bool IsPresent(string dir) => File.Exists(Path.Combine(dir, ConfigFile)) && File.Exists(Path.Combine(dir, WeightsFile));

    /// <summary>Adds the head's canonical hyper-parameters (<c>span_head.*</c>, LayerNorm epsilon 1e-5 as in torch).</summary>
    public static void AddHyperParameters(IDictionary<string, object> md, string dir)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(dir, ConfigFile)));
        var c = doc.RootElement;
        md[CanonicalKeys.SpanHeadType] = "joint_schema";
        md[CanonicalKeys.SpanHeadNormEps] = 1e-5f;
        md[CanonicalKeys.SpanHeadHeadCount] = c.GetProperty("heads").GetInt32();
        md[CanonicalKeys.SpanHeadRoutingLayerCount] = c.GetProperty("routing_layers").GetInt32();
        md[CanonicalKeys.SpanHeadLayerCount] = c.GetProperty("layers").GetInt32();
    }

    /// <summary>Opens the head file (added to <paramref name="owned"/>) and returns its tensors under canonical names.</summary>
    /// <exception cref="NotSupportedException">The head has a tensor this importer does not know, or lacks a scalar.</exception>
    public static IReadOnlyDictionary<string, WeightInfo> Weights(string dir, List<IDisposable> owned)
    {
        string path = Path.Combine(dir, WeightsFile);
        var file = SafetensorsFile.Open(path);
        owned.Add(file);
        var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        var scalars = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        foreach (var (name, tensor) in file.Tensors)
        {
            var w = tensor.ToWeightInfo();
            if (s_scalars.Contains(name))
            {
                scalars[name] = w;
            }
            else if (LayerTensor().Match(name) is { Success: true } m)
            {
                var table = m.Groups[1].Value == "evidence_layers" ? s_routing : s_joint;
                string local = table.GetValueOrDefault(m.Groups[3].Value) ?? throw Unknown(path, name);
                string prefix = $"{P}{(m.Groups[1].Value == "evidence_layers" ? "routing" : "layers")}.{m.Groups[2].Value}.{local}";
                string suffix = m.Groups[4].Value;
                if (m.Groups[3].Value.EndsWith("in_proj", StringComparison.Ordinal))
                {
                    // nn.MultiheadAttention: q, k and v stacked along the rows.
                    long rows = w.Shape[0].Fixed / 3;
                    for (int i = 0; i < 3; i++)
                    {
                        string part = $"{prefix}.{"qkv"[i]}.{suffix}";
                        weights[part] = w.RowSlice(part, i * rows, rows);
                    }
                }
                else
                {
                    weights[$"{prefix}.{suffix}"] = HfTransforms.Plain(w, $"{prefix}.{suffix}");
                }
            }
            else if (GlobalTensor().Match(name) is { Success: true } g && s_global.TryGetValue(g.Groups[1].Value, out var global))
            {
                string canonical = $"{P}{global}.{g.Groups[2].Value}";
                weights[canonical] = HfTransforms.Plain(w, canonical);
            }
            else
            {
                throw Unknown(path, name);
            }
        }

        if (scalars.Count != s_scalars.Length)
        {
            throw new NotSupportedException($"{path}: the joint schema head lacks one of {string.Join(", ", s_scalars)}.");
        }

        var values = s_scalars.Select(n => scalars[n]).ToArray();
        weights[P + "scales"] = WeightInfo.Derived(P + "scales", DataType.F32, new Shape(3), () =>
        {
            float prior = HfTransforms.DecodeF32(values[0])[0];
            float joint = HfTransforms.DecodeF32(values[1])[0];
            float gate = HfTransforms.DecodeF32(values[2])[0];
            float cap = MathF.Log(100f);
            return HfTransforms.EncodeF32([MathF.Exp(MathF.Min(prior, cap)), MathF.Exp(MathF.Min(joint, cap)), 1f / (1f + MathF.Exp(-gate))]);
        });
        return weights;
    }

    private static NotSupportedException Unknown(string path, string name) => new($"{path}: unexpected joint schema head tensor '{name}'.");
}
