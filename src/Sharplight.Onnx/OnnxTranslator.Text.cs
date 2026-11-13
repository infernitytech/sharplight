using Sharplight.Graph;

namespace Sharplight.Onnx;

// Text operators.
internal sealed partial class OnnxTranslator
{
    private void TfIdfVectorizer(OnnxNode n, TranslatedTensor?[] ins)
    {
        var ngramIndexes = n.Ints("ngram_indexes") ?? throw new InvalidDataException("TfIdfVectorizer needs ngram_indexes");
        long[] pool = n.Ints("pool_int64s") ?? throw new NotSupportedException("TfIdfVectorizer with pool_strings on integer input");
        var x = Idx(ins[0]!);
        var attributes = Attributes.From(
            ("min_gram", n.Int("min_gram_length", 1)), ("max_gram", n.Int("max_gram_length", 1)), ("max_skip", n.Int("max_skip_count", 0)),
            ("pool", pool), ("ngram_counts", n.Ints("ngram_counts") ?? [0L]), ("ngram_indexes", ngramIndexes),
            ("weights", n.Floats("weights") ?? []), ("mode", n.Str("mode", "TF")), ("out_features", ngramIndexes.Length == 0 ? 0L : ngramIndexes.Max() + 1));
        SetDyn(n, 0, Emit("TfIdf", [x], attributes, Nm(n)));
    }
}
