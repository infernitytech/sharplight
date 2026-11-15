using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Translates an OpenCLIP text tower as stored in original SD 2.x checkpoints (<c>cond_stage_model.model.*</c>: <c>resblocks</c>
/// with fused q/k/v <c>in_proj</c>, <c>c_fc</c> / <c>c_proj</c> MLPs) into transformers' <c>CLIPTextModel</c> names, which
/// <see cref="DiffusersNames.AddClipText"/> maps further.
/// </summary>
internal static partial class OpenClipNames
{
    /// <summary>
    /// The text tower's tensors under transformers names. SD 2.x conditions on the penultimate layer (followed by the final
    /// LayerNorm), so the last <paramref name="droppedLayers"/> residual blocks are left out, as diffusers' conversion does.
    /// </summary>
    /// <param name="tensors">The checkpoint's tensors.</param>
    /// <param name="prefix">The tower's prefix (<c>cond_stage_model.model.</c>).</param>
    /// <param name="droppedLayers">Trailing residual blocks not used for the conditioning.</param>
    /// <exception cref="InvalidDataException">A tensor of the tower is not recognized.</exception>
    public static IEnumerable<(string Name, WeightInfo Weight)> ToTransformers(IReadOnlyDictionary<string, WeightInfo> tensors, string prefix,
        int droppedLayers)
    {
        var names = tensors.Keys.Where(n => n.StartsWith(prefix, StringComparison.Ordinal)).ToList();
        int layers = names.Select(n => ResBlock().Match(n[prefix.Length..])).Where(m => m.Success)
            .Select(m => int.Parse(m.Groups[1].ValueSpan, System.Globalization.CultureInfo.InvariantCulture) + 1).DefaultIfEmpty(0).Max();
        int kept = Math.Max(1, layers - droppedLayers);
        foreach (string name in names)
        {
            string s = name[prefix.Length..];
            var weight = tensors[name];
            switch (s)
            {
                case "token_embedding.weight":
                    yield return ("text_model.embeddings.token_embedding.weight", weight);
                    continue;
                case "positional_embedding":
                    yield return ("text_model.embeddings.position_embedding.weight", weight);
                    continue;
                case "ln_final.weight" or "ln_final.bias":
                    yield return ("text_model.final_layer_norm." + s[(s.IndexOf('.') + 1)..], weight);
                    continue;
                case "text_projection" or "logit_scale":
                    // Contrastive head, not used for the conditioning.
                    continue;
            }

            var block = ResBlock().Match(s);
            if (!block.Success)
            {
                throw new InvalidDataException($"Unexpected OpenCLIP text tensor '{name}'.");
            }

            int layer = int.Parse(block.Groups[1].ValueSpan, System.Globalization.CultureInfo.InvariantCulture);
            if (layer >= kept)
            {
                continue;
            }

            string target = $"text_model.encoder.layers.{layer}.";
            string rest = block.Groups[2].Value;
            if (rest is "attn.in_proj_weight" or "attn.in_proj_bias")
            {
                string suffix = rest.EndsWith("weight", StringComparison.Ordinal) ? "weight" : "bias";
                long width = weight.Shape[0].Fixed / 3;
                yield return (target + "self_attn.q_proj." + suffix, weight.RowSlice(target + "self_attn.q_proj." + suffix, 0, width));
                yield return (target + "self_attn.k_proj." + suffix, weight.RowSlice(target + "self_attn.k_proj." + suffix, width, width));
                yield return (target + "self_attn.v_proj." + suffix, weight.RowSlice(target + "self_attn.v_proj." + suffix, 2 * width, width));
                continue;
            }

            int dot = rest.LastIndexOf('.');
            string module = rest[..dot] switch
            {
                "ln_1" => "layer_norm1",
                "ln_2" => "layer_norm2",
                "attn.out_proj" => "self_attn.out_proj",
                "mlp.c_fc" => "mlp.fc1",
                "mlp.c_proj" => "mlp.fc2",
                _ => throw new InvalidDataException($"Unexpected OpenCLIP text tensor '{name}'."),
            };
            yield return (target + module + rest[dot..], weight);
        }
    }

    [GeneratedRegex(@"^transformer\.resblocks\.(\d+)\.(.+)$")]
    private static partial Regex ResBlock();
}
