using Sharplight.VideoEnhancement;

namespace Sharplight.Studio.ViewModels.Video;

/// <summary>Everything one run of the Enhance page needs, read from the page when it starts.</summary>
/// <param name="Input">The video or image to enhance.</param>
/// <param name="Options">The options with the output size resolved for the input.</param>
/// <param name="Values">The parameter panel's values, for the history.</param>
internal sealed record EnhancementRun(EnhancementInput Input, VideoEnhancementOptions Options, Dictionary<string, string> Values);
