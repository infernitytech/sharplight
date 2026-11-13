using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// ONNX RoiAlign (onnxruntime semantics): each output bin averages (or takes the maximum of) bilinear samples on a regular grid inside
/// the bin; samples more than one pixel outside the feature map contribute zero.
/// </summary>
internal sealed unsafe class RoiAlignOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _rois;
    private readonly TRef _batch;
    private readonly TRef _y;
    private readonly int _samplingRatio;
    private readonly float _scale;
    private readonly bool _max;
    private readonly bool _halfPixel;
    private readonly int _c;
    private readonly int _h;
    private readonly int _w;
    private readonly int _oh;
    private readonly int _ow;
    private readonly Action<int> _body;

    public RoiAlignOp(TRef x, TRef rois, TRef batch, TRef y, int samplingRatio, float scale, bool max, bool halfPixel)
    {
        (_x, _rois, _batch, _y) = (x, rois, batch, y);
        (_samplingRatio, _scale, _max, _halfPixel) = (samplingRatio, scale, max, halfPixel);
        _c = (int)x.Dims[1];
        _h = (int)x.Dims[2];
        _w = (int)x.Dims[3];
        _oh = (int)y.Dims[2];
        _ow = (int)y.Dims[3];
        _body = Roi;
    }

    public override string Name => "RoiAlign";

    public override void Run(RunParameters p)
    {
        long rois = _rois.Dim(0, p);
        if (rois > 0)
        {
            CpuParallel.For((int)rois, _body);
        }
    }

    private void Roi(int r)
    {
        float* box = (float*)_rois.P + r * 4L;
        float offset = _halfPixel ? 0.5f : 0f;
        float x0 = box[0] * _scale - offset;
        float y0 = box[1] * _scale - offset;
        float roiW = box[2] * _scale - offset - x0;
        float roiH = box[3] * _scale - offset - y0;
        if (!_halfPixel)
        {
            roiW = MathF.Max(roiW, 1f);
            roiH = MathF.Max(roiH, 1f);
        }

        float binH = roiH / _oh;
        float binW = roiW / _ow;
        int gridH = _samplingRatio > 0 ? _samplingRatio : (int)MathF.Ceiling(roiH / _oh);
        int gridW = _samplingRatio > 0 ? _samplingRatio : (int)MathF.Ceiling(roiW / _ow);
        int count = Math.Max(gridH * gridW, 1);
        int n = ((int*)_batch.P)[r];
        float* output = (float*)_y.P + (long)r * _c * _oh * _ow;
        Span<int> index = stackalloc int[4];
        Span<float> weight = stackalloc float[4];
        for (int c = 0; c < _c; c++)
        {
            float* image = (float*)_x.P + ((long)n * _c + c) * _h * _w;
            for (int ph = 0; ph < _oh; ph++)
            {
                for (int pw = 0; pw < _ow; pw++)
                {
                    float sum = 0f;
                    float best = float.NegativeInfinity;
                    for (int iy = 0; iy < gridH; iy++)
                    {
                        float sy = y0 + ph * binH + (iy + 0.5f) * binH / gridH;
                        for (int ix = 0; ix < gridW; ix++)
                        {
                            float sx = x0 + pw * binW + (ix + 0.5f) * binW / gridW;
                            Bilinear(sy, sx, index, weight);
                            if (_max)
                            {
                                float v = MathF.Max(MathF.Max(weight[0] * image[index[0]], weight[1] * image[index[1]]),
                                    MathF.Max(weight[2] * image[index[2]], weight[3] * image[index[3]]));
                                best = MathF.Max(best, v);
                            }
                            else
                            {
                                sum += weight[0] * image[index[0]] + weight[1] * image[index[1]] + weight[2] * image[index[2]] + weight[3] * image[index[3]];
                            }
                        }
                    }

                    output[((long)c * _oh + ph) * _ow + pw] = !_max ? sum / count : gridH * gridW == 0 ? 0f : best;
                }
            }
        }
    }

    /// <summary>Corner offsets and weights of a bilinear sample; all weights are zero outside [-1, size].</summary>
    private void Bilinear(float y, float x, Span<int> index, Span<float> weight)
    {
        if (y < -1f || y > _h || x < -1f || x > _w)
        {
            index.Clear();
            weight.Clear();
            return;
        }

        y = MathF.Max(y, 0f);
        x = MathF.Max(x, 0f);
        int yLow = (int)y;
        int xLow = (int)x;
        int yHigh;
        int xHigh;
        if (yLow >= _h - 1)
        {
            yHigh = yLow = _h - 1;
            y = yLow;
        }
        else
        {
            yHigh = yLow + 1;
        }

        if (xLow >= _w - 1)
        {
            xHigh = xLow = _w - 1;
            x = xLow;
        }
        else
        {
            xHigh = xLow + 1;
        }

        float ly = y - yLow;
        float lx = x - xLow;
        float hy = 1f - ly;
        float hx = 1f - lx;
        index[0] = yLow * _w + xLow;
        index[1] = yLow * _w + xHigh;
        index[2] = yHigh * _w + xLow;
        index[3] = yHigh * _w + xHigh;
        weight[0] = hy * hx;
        weight[1] = hy * lx;
        weight[2] = ly * hx;
        weight[3] = ly * lx;
    }
}
