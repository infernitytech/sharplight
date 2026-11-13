using Sharplight.Graph;

namespace Sharplight.Onnx;

// Text operators (TfIdfVectorizer over token ids or strings).
internal sealed partial class OnnxTranslator
{
    private void TfIdfVectorizer(OnnxNode n, TranslatedTensor?[] ins)
    {
        var ngramIndexes = n.Ints("ngram_indexes") ?? throw new InvalidDataException("TfIdfVectorizer needs ngram_indexes");
        long[] pool;
        Value x;
        if (n.Ints("pool_int64s") is { } ints)
        {
            pool = ints;
            x = Idx(ins[0]!);
        }
        else
        {
            // String pools: number the distinct pool strings and map the input onto those ids (-1 for strings outside the pool).
            var strings = n.Attributes.TryGetValue("pool_strings", out var s) ? s.Strings : throw new InvalidDataException("TfIdfVectorizer needs a pool");
            var ids = new Dictionary<string, long>(StringComparer.Ordinal);
            pool = [.. strings.Select(t => ids.TryGetValue(t, out long id) ? id : ids[t] = ids.Count)];
            var keys = ids.Keys.ToArray();
            double[] values = [.. ids.Values.Select(v => (double)v)];
            x = Emit("LabelEncode", [StrValue(ins[0]!, Nm(n) + ".x")], Attributes.From(("keys", keys), ("values", values), ("default", -1.0), ("type", "i32")), Nm(n) + ".ids");
        }

        var attributes = Attributes.From(
            ("min_gram", n.Int("min_gram_length", 1)), ("max_gram", n.Int("max_gram_length", 1)), ("max_skip", n.Int("max_skip_count", 0)),
            ("pool", pool), ("ngram_counts", n.Ints("ngram_counts") ?? [0L]), ("ngram_indexes", ngramIndexes),
            ("weights", n.Floats("weights") ?? []), ("mode", n.Str("mode", "TF")), ("out_features", ngramIndexes.Length == 0 ? 0L : ngramIndexes.Max() + 1));
        SetDyn(n, 0, Emit("TfIdf", [x], attributes, Nm(n)));
    }
}
