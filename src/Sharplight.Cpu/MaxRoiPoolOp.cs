using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// Region-of-interest max pooling (Fast R-CNN / Caffe semantics): x f32 [N, C, H, W], rois f32 [R, 5] (batch, x1, y1, x2, y2) →
/// [R, C, PH, PW]. ROI corners are scaled and rounded half away from zero; every bin covers [floor(i·bin), ceil((i+1)·bin)).
/// </summary>
internal sealed unsafe class MaxRoiPoolOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _rois;
    private readonly TRef _y;
    private readonly float _scale;
    private readonly int _c;
    private readonly int _h;
    private readonly int _w;
    private readonly int _ph;
    private readonly int _pw;

    public MaxRoiPoolOp(TRef x, TRef rois, TRef y, float spatialScale)
    {
        _x = x;
        _rois = rois;
        _y = y;
        _scale = spatialScale;
        _c = (int)x.Dims[1];
        _h = (int)x.Dims[2];
        _w = (int)x.Dims[3];
        _ph = (int)y.Dims[2];
        _pw = (int)y.Dims[3];
    }

    public override string Name => "MaxRoiPool";

    public override void Run(RunParameters p)
    {
        long count = _rois.Dim(0, p);
        long batches = _x.Dim(0, p);
        CpuParallel.For((int)Math.Min(count * _c, int.MaxValue), job => Bin(job / _c, job % _c, batches));
    }

    private void Bin(long roi, int channel, long batches)
    {
        float* r = (float*)_rois.P + roi * 5;
        long batch = Math.Clamp((long)r[0], 0, batches - 1);
        int startW = (int)MathF.Round(r[1] * _scale, MidpointRounding.AwayFromZero);
        int startH = (int)MathF.Round(r[2] * _scale, MidpointRounding.AwayFromZero);
        int endW = (int)MathF.Round(r[3] * _scale, MidpointRounding.AwayFromZero);
        int endH = (int)MathF.Round(r[4] * _scale, MidpointRounding.AwayFromZero);
        float binH = (float)Math.Max(endH - startH + 1, 1) / _ph;
        float binW = (float)Math.Max(endW - startW + 1, 1) / _pw;
        float* xp = (float*)_x.P + (batch * _c + channel) * _h * _w;
        float* yp = (float*)_y.P + (roi * _c + channel) * _ph * _pw;
        for (int ph = 0; ph < _ph; ph++)
        {
            int h0 = Math.Clamp((int)MathF.Floor(ph * binH) + startH, 0, _h);
            int h1 = Math.Clamp((int)MathF.Ceiling((ph + 1) * binH) + startH, 0, _h);
            for (int pw = 0; pw < _pw; pw++)
            {
                int w0 = Math.Clamp((int)MathF.Floor(pw * binW) + startW, 0, _w);
                int w1 = Math.Clamp((int)MathF.Ceiling((pw + 1) * binW) + startW, 0, _w);
                float best = h1 <= h0 || w1 <= w0 ? 0f : float.NegativeInfinity;
                for (int h = h0; h < h1; h++)
                {
                    for (int w = w0; w < w1; w++)
                    {
                        best = Math.Max(best, xp[(long)h * _w + w]);
                    }
                }

                yp[ph * _pw + pw] = best;
            }
        }
    }
}
