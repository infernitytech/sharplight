using Sharplight.Graph;

namespace Sharplight.Architectures.VideoEnhancement;

/// <summary>
/// The frames a causal convolution slides over, without copying them into one tensor: <c>prefix</c> leading frames (the convolution's cache
/// from the previous chunk, or copies of the chunk's first frame for the first chunk) followed by the chunk's frames. A range of frames comes
/// back as the views of the parts it covers.
/// </summary>
internal sealed class PaddedFrames
{
    private readonly GraphBuilder _b;
    private readonly Value _frames;
    private readonly long _prefix;
    private readonly Value? _cache;

    /// <summary>Describes the padded frames of one chunk.</summary>
    /// <param name="b">The graph.</param>
    /// <param name="frames">The chunk [n, C, H, W].</param>
    /// <param name="prefix">Leading copies of the first frame when there is no cache (the cache brings its own frame count).</param>
    /// <param name="cache">The leading frames [prefix, C, H, W] (any float type), or <see langword="null"/> to repeat the chunk's first frame.</param>
    public PaddedFrames(GraphBuilder b, Value frames, long prefix, Value? cache)
    {
        _b = b;
        _frames = frames;
        _prefix = cache is null ? prefix : cache.Shape[0].Fixed;
        // Read once the chunk is there (not as soon as the cache exists); each range widens only the frames it takes.
        _cache = cache is null ? null : b.After(cache, frames);
    }

    /// <summary>The number of padded frames.</summary>
    public long Count => _prefix + _frames.Shape[0].Fixed;

    /// <summary>The views covering frames [start, start + count), in order.</summary>
    /// <param name="start">First padded frame.</param>
    /// <param name="count">Frames.</param>
    public List<Value> Range(long start, long count)
    {
        var parts = new List<Value>();
        long end = start + count;
        if (start < _prefix)
        {
            long leading = Math.Min(end, _prefix) - start;
            if (_cache is not null)
            {
                var cached = _b.Slice(_cache, 0, start, leading);
                parts.Add(_cache.Type == _frames.Type ? cached : _b.Emit1("Cast", [cached], Attributes.From(("type", _frames.Type.Name))));
            }
            else
            {
                var first = _b.Slice(_frames, 0, 0, 1);
                parts.AddRange(Enumerable.Repeat(first, (int)leading));
            }
        }

        if (end > _prefix)
        {
            long from = Math.Max(start, _prefix) - _prefix;
            parts.Add(_b.Slice(_frames, 0, from, end - _prefix - from));
        }

        return parts;
    }
}
