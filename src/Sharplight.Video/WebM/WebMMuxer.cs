using Sharplight.Video.Native;

namespace Sharplight.Video.WebM;

/// <summary>
/// Writes a WebM file (EBML header, Segment with Info, Tracks, Clusters of SimpleBlocks and Cues) holding one video track from compressed
/// VP8 or VP9 packets. A cluster starts at every keyframe; the file is assembled in memory so all sizes are known and no seeking is needed.
/// </summary>
internal static class WebMMuxer
{
    private const uint Ebml = 0x1A45DFA3;
    private const uint Segment = 0x18538067;
    private const uint Info = 0x1549A966;
    private const uint Tracks = 0x1654AE6B;
    private const uint Cluster = 0x1F43B675;
    private const uint Cues = 0x1C53BB6B;

    /// <summary>Muxes the packets.</summary>
    /// <param name="codec">The codec of the packets.</param>
    /// <param name="width">Frame width.</param>
    /// <param name="height">Frame height.</param>
    /// <param name="framesPerSecondNumerator">Frame rate numerator; packet timestamps count frames.</param>
    /// <param name="framesPerSecondDenominator">Frame rate denominator.</param>
    /// <param name="frameCount">Number of frames, for the duration.</param>
    /// <param name="packets">The packets in decode order; the first must be a keyframe.</param>
    public static byte[] Write(WebMCodec codec, int width, int height, int framesPerSecondNumerator, int framesPerSecondDenominator, int frameCount, IReadOnlyList<VideoPacket> packets)
    {
        long Milliseconds(long pts) => (pts * 1000L * framesPerSecondDenominator + framesPerSecondNumerator / 2) / framesPerSecondNumerator;

        byte[] header = EbmlElement.Master(
            Ebml,
            EbmlElement.UInt(0x4286, 1),
            EbmlElement.UInt(0x42F7, 1),
            EbmlElement.UInt(0x42F2, 4),
            EbmlElement.UInt(0x42F3, 8),
            EbmlElement.String(0x4282, "webm"),
            EbmlElement.UInt(0x4287, 4),
            EbmlElement.UInt(0x4285, 2));
        byte[] info = EbmlElement.Master(
            Info,
            EbmlElement.UInt(0x2AD7B1, 1_000_000),
            EbmlElement.String(0x4D80, "Sharplight"),
            EbmlElement.String(0x5741, "Sharplight"),
            EbmlElement.Float(0x4489, Milliseconds(frameCount)));
        long defaultDurationNanoseconds = 1_000_000_000L * framesPerSecondDenominator / framesPerSecondNumerator;
        byte[] tracks = EbmlElement.Master(
            Tracks,
            EbmlElement.Master(
                0xAE,
                EbmlElement.UInt(0xD7, 1),
                EbmlElement.UInt(0x73C5, 1),
                EbmlElement.UInt(0x83, 1),
                EbmlElement.UInt(0x9C, 0),
                EbmlElement.String(0x86, codec == WebMCodec.Vp9 ? "V_VP9" : "V_VP8"),
                EbmlElement.UInt(0x23E383, (ulong)defaultDurationNanoseconds),
                EbmlElement.Master(0xE0, EbmlElement.UInt(0xB0, (ulong)width), EbmlElement.UInt(0xBA, (ulong)height))));

        // Clusters are built first so the cue positions (relative to the start of the segment payload) are known.
        long segmentOffset = info.Length + tracks.Length;
        var clusters = new List<byte[]>();
        var cuePoints = new List<byte[]>();
        var blocks = new List<byte[]>();
        long clusterStart = 0;
        long clusterTimecode = 0;
        for (int i = 0; i < packets.Count; i++)
        {
            VideoPacket packet = packets[i];
            long timecode = Milliseconds(packet.Pts);
            if (blocks.Count > 0 && (packet.IsKeyframe || timecode - clusterTimecode > short.MaxValue))
            {
                segmentOffset += FlushCluster(clusters, blocks, clusterTimecode);
            }

            if (blocks.Count == 0)
            {
                clusterStart = segmentOffset;
                clusterTimecode = timecode;
                if (packet.IsKeyframe)
                {
                    cuePoints.Add(EbmlElement.Master(
                        0xBB,
                        EbmlElement.UInt(0xB3, (ulong)timecode),
                        EbmlElement.Master(0xB7, EbmlElement.UInt(0xF7, 1), EbmlElement.UInt(0xF1, (ulong)clusterStart))));
                }
            }

            blocks.Add(SimpleBlock(packet, (short)(timecode - clusterTimecode)));
        }

        FlushCluster(clusters, blocks, clusterTimecode);
        byte[] cues = EbmlElement.Master(Cues, [.. cuePoints]);
        byte[][] segmentChildren = [info, tracks, .. clusters, cues];
        byte[] segment = EbmlElement.Master(Segment, segmentChildren);
        var file = new byte[header.Length + segment.Length];
        header.CopyTo(file, 0);
        segment.CopyTo(file, header.Length);
        return file;
    }

    private static long FlushCluster(List<byte[]> clusters, List<byte[]> blocks, long clusterTimecode)
    {
        byte[] cluster = EbmlElement.Master(Cluster, [EbmlElement.UInt(0xE7, (ulong)clusterTimecode), .. blocks]);
        clusters.Add(cluster);
        blocks.Clear();
        return cluster.Length;
    }

    private static byte[] SimpleBlock(VideoPacket packet, short relativeTimecode)
    {
        var payload = new byte[4 + packet.Data.Length];
        payload[0] = 0x81;
        payload[1] = (byte)(relativeTimecode >> 8);
        payload[2] = (byte)relativeTimecode;
        payload[3] = packet.IsKeyframe ? (byte)0x80 : (byte)0;
        packet.Data.CopyTo(payload, 4);
        return EbmlElement.Binary(0xA3, payload);
    }
}
