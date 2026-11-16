using Sharplight.Samples;
using Sharplight.Vision.Sample;

var options = new VisionOptionSet();
var command = options.BuildCommand(VisionApplication.ClassifyAsync, VisionApplication.DetectAsync, VisionApplication.DepthAsync, VisionApplication.SegmentAsync, VisionApplication.OcrAsync,
    VisionApplication.UpscaleAsync);
return await CommandRunner.RunAsync(command, args);
