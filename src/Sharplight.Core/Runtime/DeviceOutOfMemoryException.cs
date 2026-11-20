namespace Sharplight.Runtime;

/// <summary>
/// A device could not allocate memory, after every fallback the provider has (other memory types, smaller blocks, smaller staging
/// buffers). The message names the allocation, its size and the device's memory figures; hosts can catch it to offload to another
/// device, lower the context, or report it.
/// </summary>
public sealed class DeviceOutOfMemoryException : OutOfMemoryException
{
    /// <summary>Creates the exception.</summary>
    /// <param name="device">The device's name.</param>
    /// <param name="what">What was being allocated (e.g. "weights", "upload staging").</param>
    /// <param name="requestedBytes">The size that could not be allocated.</param>
    /// <param name="details">The device's memory figures and the attempts made, for the message.</param>
    /// <param name="innerException">The underlying error, if any.</param>
    public DeviceOutOfMemoryException(string device, string what, long requestedBytes, string details, Exception? innerException = null)
        : base(Format(device, what, requestedBytes, details), innerException)
    {
        Device = device;
        What = what;
        RequestedBytes = requestedBytes;
    }

    /// <summary>The device's name.</summary>
    public string Device { get; }

    /// <summary>What was being allocated.</summary>
    public string What { get; }

    /// <summary>The size that could not be allocated, in bytes.</summary>
    public long RequestedBytes { get; }

    private static string Format(string device, string what, long requestedBytes, string details)
        => $"{device}: out of memory allocating {requestedBytes / 1048576.0:0.0} MiB for {what}. {details}"
            + " Free memory on the device (other models or applications), lower the context length or batch size, or place part of the model on another device.";
}
