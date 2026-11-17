using System.Runtime.InteropServices;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using Avalonia.Threading;
using Sharplight.Vision;

namespace Sharplight.Studio.Views;

/// <summary>
/// Shows the current frame of a video as an ordinary Avalonia control (an <see cref="ImageViewer"/>: fit, zoom, pan), so it clips,
/// scrolls and stays below dialogs and flyouts like any other control. Each frame is converted to a bitmap on the thread pool into one of
/// two <see cref="WriteableBitmap"/>s while the other is on screen; the UI thread only swaps the source. A frame that arrives while
/// another is converting replaces the waiting one, so a slow machine drops frames instead of stalling the UI.
/// </summary>
internal sealed class VideoView : UserControl
{
    private const int BytesPerPixel = 4;

    /// <summary>The frame shown (8-bit RGB), or null.</summary>
    public static readonly StyledProperty<ImageInput?> FrameProperty = AvaloniaProperty.Register<VideoView, ImageInput?>(nameof(Frame));

    private readonly ImageViewer _viewer = new();
    private readonly WriteableBitmap?[] _buffers = new WriteableBitmap?[2];
    private readonly object _gate = new();
    private ImageInput? _waiting;
    private bool _converting;
    private int _next;
    private TaskCompletionSource _idle = Completed();

    /// <summary>Creates the view.</summary>
    public VideoView() => Content = _viewer;

    /// <summary>The frame shown.</summary>
    public ImageInput? Frame
    {
        get => GetValue(FrameProperty);
        set => SetValue(FrameProperty, value);
    }

    /// <summary>The viewer inside (zoom state).</summary>
    public ImageViewer Viewer => _viewer;

    /// <summary>Completes when every frame handed over so far was converted and set as the viewer's source (for tests).</summary>
    public Task Idle
    {
        get
        {
            lock (_gate)
            {
                return _idle.Task;
            }
        }
    }

    /// <inheritdoc/>
    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
    {
        base.OnPropertyChanged(change);
        if (change.Property == FrameProperty)
        {
            Submit(change.GetNewValue<ImageInput?>());
        }
    }

    private static TaskCompletionSource Completed()
    {
        var source = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        source.SetResult();
        return source;
    }

    private void Submit(ImageInput? frame)
    {
        if (frame is not { Width: > 0, Height: > 0 })
        {
            lock (_gate)
            {
                _waiting = null;
            }

            _viewer.Source = null;
            return;
        }

        lock (_gate)
        {
            _waiting = frame;
            if (_converting)
            {
                return;
            }

            _converting = true;
            _idle = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        }

        _ = Task.Run(ConvertLoop);
    }

    private void ConvertLoop()
    {
        while (true)
        {
            ImageInput frame;
            lock (_gate)
            {
                if (_waiting is null)
                {
                    _converting = false;
                    _idle.TrySetResult();
                    return;
                }

                frame = _waiting;
                _waiting = null;
            }

            WriteableBitmap bitmap;
            try
            {
                bitmap = Convert(frame);
            }
            catch (Exception error) when (error is InvalidOperationException or ArgumentException)
            {
                continue;
            }

            Dispatcher.UIThread.Post(() => _viewer.Source = bitmap, DispatcherPriority.Render);
        }
    }

    private WriteableBitmap Convert(ImageInput frame)
    {
        var size = new PixelSize(frame.Width, frame.Height);
        int slot = _next;
        _next ^= 1;
        var bitmap = _buffers[slot];
        if (bitmap is null || bitmap.PixelSize != size)
        {
            bitmap = new WriteableBitmap(size, new Vector(96, 96), PixelFormat.Bgra8888, AlphaFormat.Opaque);
            _buffers[slot] = bitmap;
        }

        using var buffer = bitmap.Lock();
        var row = new byte[frame.Width * BytesPerPixel];
        byte[] rgb = frame.Rgb;
        for (int y = 0; y < frame.Height; y++)
        {
            int source = y * frame.Width * 3;
            for (int x = 0; x < frame.Width; x++)
            {
                int target = x * BytesPerPixel;
                row[target] = rgb[source + 2];
                row[target + 1] = rgb[source + 1];
                row[target + 2] = rgb[source];
                row[target + 3] = 255;
                source += 3;
            }

            Marshal.Copy(row, 0, buffer.Address + y * buffer.RowBytes, row.Length);
        }

        return bitmap;
    }
}
