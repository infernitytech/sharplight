using System.Runtime.InteropServices;
using Microsoft.Extensions.Options;
using Sharplight.Audio.Native;

namespace Sharplight.Audio;

/// <summary>
/// <see cref="IAudioDeviceProvider"/> over the native library (WASAPI, Core Audio, PulseAudio, ALSA, JACK and others, loaded at run
/// time). The backend is initialized on first use; a machine without audio support reports <see cref="IsAvailable"/> false.
/// </summary>
public sealed unsafe class NativeAudioDeviceProvider : IAudioDeviceProvider, IDisposable
{
    private const int NameBufferSize = 512;

    private readonly NativeAudioDeviceOptions _options;
    private readonly Lock _gate = new();
    private NativeContextHandle? _context;
    private string _description = string.Empty;
    private bool _initialized;
    private bool _disposed;

    /// <summary>Creates the provider.</summary>
    /// <param name="options">Settings; <see langword="null"/> selects the defaults.</param>
    public NativeAudioDeviceProvider(IOptions<NativeAudioDeviceOptions>? options = null)
    {
        _options = options?.Value ?? new NativeAudioDeviceOptions();
    }

    /// <inheritdoc />
    public bool IsAvailable => GetContext() is not null;

    /// <inheritdoc />
    public string Description
    {
        get
        {
            GetContext();
            return _description;
        }
    }

    /// <inheritdoc />
    public IReadOnlyList<AudioDeviceInfo> GetDevices(AudioDeviceKind kind)
    {
        NativeContextHandle? context = GetContext();
        if (context is null)
        {
            return [];
        }

        using HandleLease lease = context.Lease();
        int result = NativeMethods.sla_device_count(lease.Value, (int)kind, out int count);
        if (result < 0)
        {
            return [];
        }

        var devices = new List<AudioDeviceInfo>(count);
        byte* buffer = stackalloc byte[NameBufferSize];
        for (int index = 0; index < count; index++)
        {
            result = NativeMethods.sla_device_info(lease.Value, (int)kind, index, buffer, NameBufferSize, out int isDefault);
            if (result < 0 && result != (int)NativeResult.BufferTooSmall)
            {
                continue;
            }

            devices.Add(new AudioDeviceInfo(kind, Marshal.PtrToStringUTF8((nint)buffer) ?? string.Empty, isDefault != 0));
        }

        return devices;
    }

    /// <inheritdoc />
    public IAudioOutput OpenOutput(AudioStreamOptions? options = null)
    {
        (NativeContextHandle context, NativeStreamHandle stream, int sampleRate, int channels, int ringFrames) = Open(AudioDeviceKind.Playback, options);
        return new NativeAudioOutput(context, stream, sampleRate, channels, ringFrames);
    }

    /// <inheritdoc />
    public IAudioInput OpenInput(AudioStreamOptions? options = null)
    {
        (NativeContextHandle context, NativeStreamHandle stream, int sampleRate, int channels, int _) = Open(AudioDeviceKind.Capture, options);
        return new NativeAudioInput(context, stream, sampleRate, channels);
    }

    /// <summary>Closes the backend; streams that are still open keep it alive until they are disposed.</summary>
    public void Dispose()
    {
        lock (_gate)
        {
            _disposed = true;
            _context?.Dispose();
            _context = null;
        }
    }

    private static int RingFrames(int requested) => (int)System.Numerics.BitOperations.RoundUpToPowerOf2((uint)Math.Max(requested, 64));

    private (NativeContextHandle Context, NativeStreamHandle Stream, int SampleRate, int Channels, int RingFrames) Open(AudioDeviceKind kind, AudioStreamOptions? options)
    {
        options ??= new AudioStreamOptions();
        if (options.SampleRate < 8000 || options.Channels < 1)
        {
            throw new ArgumentException("The stream needs a sample rate of at least 8000 Hz and at least one channel.", nameof(options));
        }

        NativeContextHandle context = GetContext() ?? throw new InvalidOperationException($"No audio device is available: {Description}");
        int deviceIndex = -1;
        if (options.DeviceName is not null)
        {
            IReadOnlyList<AudioDeviceInfo> devices = GetDevices(kind);
            deviceIndex = devices.ToList().FindIndex(d => string.Equals(d.Name, options.DeviceName, StringComparison.Ordinal));
            if (deviceIndex < 0)
            {
                throw new InvalidOperationException($"The audio device '{options.DeviceName}' was not found.");
            }
        }

        int ringFrames = RingFrames((int)(options.BufferDuration.TotalSeconds * options.SampleRate));
        bool added = false;
        context.DangerousAddRef(ref added);
        try
        {
            int result = NativeMethods.sla_stream_open(context.DangerousGetHandle(), (int)kind, deviceIndex, (uint)options.SampleRate, (uint)options.Channels, (uint)ringFrames, out nint stream);
            if (result == (int)NativeResult.Device)
            {
                throw new InvalidOperationException("The audio device could not be opened.");
            }

            NativeException.ThrowIfFailed(result, "Opening the audio device");
            return (context, new NativeStreamHandle(stream), options.SampleRate, options.Channels, ringFrames);
        }
        catch
        {
            context.DangerousRelease();
            throw;
        }
    }

    private NativeContextHandle? GetContext()
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_initialized)
            {
                return _context;
            }

            _initialized = true;
            if (!NativeAudioLibrary.IsAvailable)
            {
                _description = NativeAudioLibrary.UnavailableReason ?? "The native audio library is unavailable.";
                return null;
            }

            int result = NativeMethods.sla_context_create(_options.UseNullBackend ? 1 : 0, out nint context);
            if (result < 0)
            {
                _description = "No audio backend is available on this machine (no sound server or audio driver was found).";
                return null;
            }

            _context = new NativeContextHandle(context);
            _description = Marshal.PtrToStringUTF8(NativeMethods.sla_context_backend_name(context)) ?? "audio";
            return _context;
        }
    }
}
