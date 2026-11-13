using Sharplight.Graph;

namespace Sharplight.Onnx;

// ai.onnx.ml linear models and support vector machines: scores are MatMuls against constant coefficient matrices (and support vectors),
// so they run on every provider; the one-vs-one vote of SVM classifiers is the IR op OneVsOne.
internal sealed partial class OnnxTranslator
{
    private void LinearRegressor(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        int targets = (int)n.Int("targets", 1);
        var y = Affine(x, n.Floats("coefficients") ?? [], targets, n.Floats("intercepts"), Nm(n));
        SetDyn(n, 0, PostTransform(y, n.Str("post_transform", "NONE"), Nm(n)));
    }

    /// <summary>
    /// Scores X·Wᵀ + b, then the post-transform; a single score column with two classes is binary (onnxruntime: scores [1 - t, t] of the
    /// transformed score t, class 1 when the raw score is positive, or t &gt; 0.5 after a transform).
    /// </summary>
    private void LinearClassifier(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        string name = Nm(n);
        var coefficients = n.Floats("coefficients") ?? throw new InvalidDataException("LinearClassifier needs coefficients");
        long features = x.Shape[-1].Fixed;
        int columns = (int)(coefficients.Length / features);
        string transform = n.Str("post_transform", "NONE");
        var scores = Affine(x, coefficients, columns, n.Floats("intercepts"), name + ".raw");
        if (columns == 1 && ClassCount(n) == 2)
        {
            var t = PostTransform(scores, transform, name);
            var positive = transform == "NONE" ? Emit("Greater", [scores, F32Const([0f], name + ".zero")]) : Emit("Greater", [t, F32Const([0.5f], name + ".half")]);
            SetDyn(n, 0, ClassLabels(n, Rs(Emit("Cast", [positive], Attributes.From(("type", "i32"))), Fixed(x)[0]), name + ".label"));
            SetDyn(n, 1, Concat2(_b.Scale(t, -1f, 1f), t));
            return;
        }

        var transformed = PostTransform(scores, transform, name);
        SetDyn(n, 0, ClassLabels(n, ArgMaxRows(transformed, name), name + ".label"));
        SetDyn(n, 1, transformed);
    }

    private void SvmRegressor(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        string name = Nm(n);
        var coefficients = n.Floats("coefficients") ?? throw new InvalidDataException("SVMRegressor needs coefficients");
        float rho = (n.Floats("rho") ?? [0f])[0];
        Value y;
        if (n.Int("n_supports", 0) == 0)
        {
            y = Affine(x, coefficients, 1, [rho], name + ".raw");
        }
        else
        {
            var kernel = SvmKernel(n, x, name);
            y = Affine(kernel, coefficients, 1, [rho], name + ".raw");
        }

        if (n.Int("one_class", 0) != 0)
        {
            var positive = Emit("Greater", [y, F32Const([0f], name + ".zero")]);
            y = _b.Scale(positive, 2f, -1f);
        }

        SetDyn(n, 0, PostTransform(y, n.Str("post_transform", "NONE"), name));
    }

    /// <summary>
    /// libsvm one-vs-one classifier: pairwise decisions are one MatMul over the kernel values; without support vectors (linear mode) one
    /// score per class (largest wins). Scores are the decisions ([-d, d] for two classes) or the coupled probabilities.
    /// </summary>
    private void SvmClassifier(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        string name = Nm(n);
        var coefficients = n.Floats("coefficients") ?? throw new InvalidDataException("SVMClassifier needs coefficients");
        var rho = n.Floats("rho") ?? throw new InvalidDataException("SVMClassifier needs rho");
        string transform = n.Str("post_transform", "NONE");
        int classes = ClassCount(n);
        if (n.Ints("vectors_per_class") is not { Length: > 0 } perClass)
        {
            // onnxruntime adds rho[0] to every class score in linear mode.
            var linear = Affine(x, coefficients, coefficients.Length / (int)x.Shape[-1].Fixed, [rho[0]], name + ".raw");
            var transformed = PostTransform(linear, transform, name);
            SetDyn(n, 0, ClassLabels(n, ArgMaxRows(transformed, name), name + ".label"));
            SetDyn(n, 1, transformed);
            return;
        }

        var kernel = SvmKernel(n, x, name);
        int vectors = (int)perClass.Sum();
        var starts = new int[classes + 1];
        for (int c = 0; c < classes; c++)
        {
            starts[c + 1] = starts[c] + (int)perClass[c];
        }

        // Pair (i, j) sums class i's vectors with coefficient row j - 1 and class j's vectors with row i.
        int pairs = classes * (classes - 1) / 2;
        var pairWeights = new float[pairs * vectors];
        int pair = 0;
        for (int i = 0; i < classes; i++)
        {
            for (int j = i + 1; j < classes; j++, pair++)
            {
                for (int v = starts[i]; v < starts[i + 1]; v++)
                {
                    pairWeights[(pair * vectors) + v] = coefficients[((j - 1) * vectors) + v];
                }

                for (int v = starts[j]; v < starts[j + 1]; v++)
                {
                    pairWeights[(pair * vectors) + v] = coefficients[(i * vectors) + v];
                }
            }
        }

        var decisions = Affine(kernel, pairWeights, pairs, rho, name + ".decisions");
        var attributes = Attributes.From(("classes", (long)classes));
        bool probabilities = n.Floats("prob_a") is { Length: > 0 };
        if (probabilities)
        {
            attributes = attributes.With("prob_a", n.Floats("prob_a")!).With("prob_b", n.Floats("prob_b") ?? new float[pairs]);
        }

        var vote = _b.Emit("OneVsOne", [decisions], attributes, name + ".vote");
        SetDyn(n, 0, ClassLabels(n, vote[0], name + ".label"));
        var scores = probabilities ? vote[1] : classes == 2 ? Concat2(_b.Scale(decisions, -1f), decisions) : decisions;
        SetDyn(n, 1, PostTransform(scores, transform, name));
    }

    /// <summary>Kernel values [N, vectors] between the rows of x and the support vectors.</summary>
    private Value SvmKernel(OnnxNode n, Value x, string name)
    {
        var supportVectors = n.Floats("support_vectors") ?? throw new InvalidDataException($"{n.OpType} needs support_vectors");
        long features = x.Shape[-1].Fixed;
        int vectors = (int)(supportVectors.Length / features);
        var parameters = n.Floats("kernel_params") ?? [0f, 0f, 0f];
        float gamma = parameters.Length > 0 ? parameters[0] : 0f;
        float coef0 = parameters.Length > 1 ? parameters[1] : 0f;
        float degree = parameters.Length > 2 ? parameters[2] : 0f;
        var dot = Affine(x, supportVectors, vectors, null, name + ".dot");
        switch (n.Str("kernel_type", "LINEAR"))
        {
            case "LINEAR":
                return dot;
            case "POLY":
                return Emit("Pow", [_b.Scale(dot, gamma, coef0), F32Const([degree], name + ".degree")], null, name + ".kernel");
            case "SIGMOID":
                return Emit("Tanh", [_b.Scale(dot, gamma, coef0)], null, name + ".kernel");
            case "RBF":
                {
                    // exp(-gamma · |x - v|²) with |x - v|² = |x|² - 2 x·v + |v|².
                    var squares = new float[vectors];
                    for (int v = 0; v < vectors; v++)
                    {
                        for (int f = 0; f < features; f++)
                        {
                            squares[v] += supportVectors[(v * features) + f] * supportVectors[(v * features) + f];
                        }
                    }

                    var xx = Emit("Reduce", [x], Attributes.From(("mode", "sumsquare"), ("axes", new[] { -1L }), ("keepdims", true)), name + ".xx");
                    var distance = Emit("Add", [Emit("Add", [_b.Scale(dot, -2f), xx]), F32Const(squares, name + ".vv")]);
                    var clamped = Emit("Max", [distance, F32Const([0f], name + ".zero")]);
                    return Emit("Exp", [_b.Scale(clamped, -gamma)], null, name + ".kernel");
                }

            default:
                throw new NotSupportedException($"{n.OpType} kernel_type '{n.Str("kernel_type", "")}'");
        }
    }

    /// <summary>x [N, F] · Wᵀ + b with W the row-major [outputs, F] coefficients (an IR MatMul weight) and b per output (or broadcast).</summary>
    private Value Affine(Value x, float[] coefficients, int outputs, float[]? bias, string name)
    {
        long features = x.Shape[-1].Fixed;
        if (coefficients.Length != outputs * features)
        {
            throw new InvalidDataException($"{name}: {coefficients.Length} coefficients for {outputs} outputs of {features} features");
        }

        var weight = OnnxConst.FromBytes(OnnxType.Float, [outputs, features], ToBytes(coefficients), name + ".weight");
        var y = Emit("MatMul", [x, WeightValue(weight, keepHalf: false, name + ".weight")], null, name);
        Heavy();
        return bias is { Length: > 0 } ? Emit("Add", [y, F32Const(bias, name + ".bias")], null, name + ".biased") : y;
    }

    private Value Concat2(Value a, Value b) => _b.Concat(a.Rank - 1, a, b);
}
