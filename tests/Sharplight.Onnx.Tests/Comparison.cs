namespace Sharplight.Onnx.Tests;

internal readonly record struct Comparison(double MaxAbs, double MaxRel, double Scale, bool Ok, int Worst, double RmsRel = 0);
