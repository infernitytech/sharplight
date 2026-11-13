using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// ONNX NonMaxSuppression (onnxruntime semantics): per batch and class, greedily keeps the highest-scoring boxes (ties: lower index first)
/// whose score exceeds the threshold and whose IoU with every kept box is at most the IoU threshold. Writes i32 rows (batch, class, box) and
/// publishes their number as the dynamic symbol of the output.
/// </summary>
internal sealed unsafe class NonMaxSuppressionOp : CpuOp
{
    private readonly TRef _boxes;
    private readonly TRef _scores;
    private readonly TRef _y;
    private readonly string _countSymbol;
    private readonly long _maxPerClass;
    private readonly float _iouThreshold;
    private readonly float _scoreThreshold;
    private readonly bool _centerPointBox;

    public NonMaxSuppressionOp(TRef boxes, TRef scores, TRef y, string countSymbol, long maxPerClass, float iouThreshold, float scoreThreshold, bool centerPointBox)
    {
        _boxes = boxes;
        _scores = scores;
        _y = y;
        _countSymbol = countSymbol;
        _maxPerClass = maxPerClass;
        _iouThreshold = iouThreshold;
        _scoreThreshold = scoreThreshold;
        _centerPointBox = centerPointBox;
    }

    public override string Name => "NonMaxSuppression";

    public override void Run(RunParameters p)
    {
        long batches = _scores.Dim(0, p);
        long classes = _scores.Dims[1];
        long boxCount = _scores.Dims[2];
        float* boxes = (float*)_boxes.P;
        float* scores = (float*)_scores.P;
        int* output = (int*)_y.P;
        long count = 0;
        var candidates = new List<int>();
        var kept = new List<int>();
        for (long b = 0; b < batches; b++)
        {
            float* batchBoxes = boxes + b * boxCount * 4;
            for (long c = 0; c < classes && _maxPerClass > 0; c++)
            {
                float* classScores = scores + (b * classes + c) * boxCount;
                candidates.Clear();
                for (int i = 0; i < boxCount; i++)
                {
                    if (classScores[i] > _scoreThreshold)
                    {
                        candidates.Add(i);
                    }
                }

                candidates.Sort((u, v) => classScores[u] != classScores[v] ? classScores[v].CompareTo(classScores[u]) : u.CompareTo(v));
                kept.Clear();
                foreach (int candidate in candidates)
                {
                    if (kept.Count >= _maxPerClass)
                    {
                        break;
                    }

                    if (kept.All(k => !Suppresses(batchBoxes + k * 4, batchBoxes + candidate * 4)))
                    {
                        kept.Add(candidate);
                        output[count * 3] = (int)b;
                        output[count * 3 + 1] = (int)c;
                        output[count * 3 + 2] = candidate;
                        count++;
                    }
                }
            }
        }

        p.Symbols[_countSymbol] = count;
    }

    private bool Suppresses(float* a, float* b)
    {
        var (ay1, ax1, ay2, ax2) = Corners(a);
        var (by1, bx1, by2, bx2) = Corners(b);
        float width = Math.Min(ax2, bx2) - Math.Max(ax1, bx1);
        float height = Math.Min(ay2, by2) - Math.Max(ay1, by1);
        if (width <= 0 || height <= 0)
        {
            return false;
        }

        float intersection = width * height;
        float areaA = (ax2 - ax1) * (ay2 - ay1);
        float areaB = (bx2 - bx1) * (by2 - by1);
        float union = areaA + areaB - intersection;
        if (areaA <= 0 || areaB <= 0 || union <= 0)
        {
            return false;
        }

        return intersection / union > _iouThreshold;
    }

    /// <summary>(y1, x1, y2, x2) with y1 ≤ y2 and x1 ≤ x2.</summary>
    private (float Y1, float X1, float Y2, float X2) Corners(float* box)
    {
        if (_centerPointBox)
        {
            float halfWidth = box[2] / 2;
            float halfHeight = box[3] / 2;
            return (box[1] - halfHeight, box[0] - halfWidth, box[1] + halfHeight, box[0] + halfWidth);
        }

        return (Math.Min(box[0], box[2]), Math.Min(box[1], box[3]), Math.Max(box[0], box[2]), Math.Max(box[1], box[3]));
    }
}
