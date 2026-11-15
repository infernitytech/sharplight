namespace Sharplight.Vulkan;

/// <summary>
/// The stored device code of one kernel (VK_KHR_pipeline_binary): the key of the pipeline it was captured from and the driver
/// binaries that make up that pipeline, each with its binary key. See <see cref="PipelineBinaryCodec"/>.
/// </summary>
/// <param name="PipelineKey">vkGetPipelineKeyKHR of the create info the binaries were captured with; a load uses them only when its
/// own create info has the same key.</param>
/// <param name="Binaries">The binaries: key and data.</param>
internal sealed record PipelineBinaryEntry(byte[] PipelineKey, IReadOnlyList<(byte[] Key, byte[] Data)> Binaries);
