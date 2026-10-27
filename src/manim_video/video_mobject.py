from collections.abc import Iterator
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import override

import av
import numpy as np
from manim import (
    DL,
    UL,
    UR,
    Group,
    ImageMobject,
    Scene,
    SurroundingRectangle,
    change_to_rgba_array,
)
from PIL import Image, ImageOps
from pydub import AudioSegment


@dataclass
class VideoStatus:
    time: float = 0
    container: av.container.InputContainer | None = None
    stream: av.video.stream.VideoStream | None = None
    frames: Iterator[av.VideoFrame] | None = None
    current_frame: av.VideoFrame | None = None
    next_frame: av.VideoFrame | None = None
    exhausted: bool = False
    closed: bool = False

    # Share source audio and its open interval with copied mobjects
    audio: AudioSegment | None = None
    audio_recording: bool = False
    audio_scene_start: float = 0
    audio_source_start: float = 0
    audio_speed: float = 1

    def __deepcopy__(self, memo):
        # Keep copied mobjects attached to the same decoder and playback clock
        return self


class VideoMobject(ImageMobject):
    """Play a video while preserving its alpha channel.

    Parameters
    ----------
    file_path
        Path to the source video
    scene
        Scene whose clock and audio timeline are used for playback
    imageops
        Optional PIL image operation applied to each RGBA frame
        Operations that convert to RGB will discard transparency
    silent
        Skip audio loading and playback when True
        Sources without an audio track automatically play silently
    speed
        Nonnegative playback multiplier, with 0 pausing playback
    loop
        Repeat the selected interval from start_time
    start_time
        Start position in source video seconds, defaults to 0
    end_time
        Exclusive end position in source video seconds
        None uses the natural video end
        Non-looping playback holds the last displayed frame before the end
    close_at_end
        Close the decoder after non-looping playback finishes
        Defaults to True and leaves the final image visible
        Looping playback keeps the decoder open

    Notes
    -----
    remaining_time() returns seconds until the current pass ends
    close() releases the decoder and stops this object's updater
    Copies share playback state and the decoder

    Call vid.close() whenever you remove the video from the scene,
    including after FadeOut. Manim does not notify mobjects when
    they are removed. Also close remaining videos before the scene ends.

    Adapted from the original OpenCV VideoMobject by
    Uwe Zimmermann and Abulafia
    """

    def __init__(
        self,
        file_path: str | Path,
        scene: Scene,
        imageops=None,
        silent=False,
        speed=1.0,
        loop=False,
        start_time=0.0,
        end_time=None,
        close_at_end: bool = True,
        **kwargs,
    ):
        # Validate the interval and forward playback speed
        if not isfinite(start_time) or start_time < 0:
            raise ValueError("start_time must be finite and nonnegative")
        if end_time is not None and (not isfinite(end_time) or end_time <= start_time):
            raise ValueError("end_time must be finite and greater than start_time")
        if not isfinite(speed) or speed < 0:
            raise ValueError("speed must be finite and nonnegative")

        self.file_path = Path(file_path)
        self.imageops = imageops
        self._speed = speed
        self.loop = loop
        self.close_at_end = close_at_end
        self.start_ms = start_time * 1000
        self.end_ms = None if end_time is None else end_time * 1000
        self.status = VideoStatus(time=self.start_ms)
        self.silent = silent
        self.scene = scene

        # Open the decoder and release it if initialization fails
        container = av.open(file_path)
        self.status.container = container
        try:
            if not container.streams.video:
                raise ValueError("The file contains no video stream")

            # Play silently if the source has no audio track
            if not container.streams.audio:
                self.silent = True

            # Load source audio once for all playback intervals
            if not self.silent:
                self.status.audio = AudioSegment.from_file(self.file_path)

            stream = container.streams.video[0]
            self.status.stream = stream
            self._stream_start = stream.start_time or 0

            # Prefer video duration over the duration of other container streams
            self.duration_ms = None
            if stream.duration is not None:
                self.duration_ms = float(stream.duration * stream.time_base) * 1000
            elif container.duration is not None:
                self.duration_ms = container.duration / av.time_base * 1000

            # Stop at the natural video end if it precedes the requested end
            self._playback_end_ms = self.end_ms
            if self.duration_ms is not None:
                if self._playback_end_ms is None:
                    self._playback_end_ms = self.duration_ms
                else:
                    self._playback_end_ms = min(self._playback_end_ms, self.duration_ms)
                if self.start_ms >= self._playback_end_ms:
                    raise ValueError("start_time is beyond the video end")

            # Decode the initial image at the selected start
            frame = self._read_frame(self.start_ms, seek=True)
            if frame is None:
                raise ValueError("No video frame is available at start_time")

            super().__init__(self._frame_image(frame), **kwargs)
        except Exception:
            container.close()
            self.status.closed = True
            raise

        # Advance playback whenever Manim updates this mobject
        self.add_updater(self.video_updater)

    def __deepcopy__(self, memo):
        # Keep copied videos attached to the original scene
        memo[id(self.scene)] = self.scene
        return super().__deepcopy__(memo)

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, value: float):
        # Validate before changing playback or committing audio
        if not isfinite(value) or value < 0:
            raise ValueError("speed must be finite and nonnegative")

        # Leave an unchanged speed and its audio interval alone
        if value == self._speed:
            return

        # Commit audio at the old speed before changing it
        self._stop_audio()
        self._speed = value

        # Start another interval if playback is active
        if not self.updating_suspended and not self.status.closed:
            self._start_audio()

    def _frame_time_ms(self, frame):
        # Normalize source timestamps so playback starts at video time zero
        if frame.pts is None:
            raise ValueError("The video frame has no presentation timestamp")
        start_seconds = float(self._stream_start * self.status.stream.time_base)
        return (float(frame.pts * frame.time_base) - start_seconds) * 1000

    def _next_frame(self):
        # Mark the natural end when the decoder runs out of frames
        frame = next(self.status.frames, None)
        if frame is None:
            self.status.exhausted = True
        return frame

    def _read_frame(self, time_ms, *, seek=False):
        status = self.status
        stream = status.stream

        # Seek to a preceding keyframe when starting or wrapping playback
        if seek:
            timestamp = self._stream_start + int(time_ms / 1000 / stream.time_base)
            status.container.seek(timestamp, stream=stream, backward=True)
            status.frames = iter(status.container.decode(stream))
            status.current_frame = None
            status.next_frame = None
            status.exhausted = False

        # Keep one frame ahead so the current image covers the requested time
        if status.next_frame is None and not status.exhausted:
            status.next_frame = self._next_frame()

        while (
            status.next_frame is not None
            and self._frame_time_ms(status.next_frame) <= time_ms
        ):
            status.current_frame = status.next_frame
            status.next_frame = self._next_frame()

        return status.current_frame

    def _frame_image(self, frame):
        # Preserve the decoded alpha channel all the way into the PIL image
        img = Image.fromarray(frame.to_ndarray(format="rgba"))
        if self.imageops is not None:
            img = self.imageops(img)
        return img

    def remaining_time(self) -> float:
        """Return playback seconds until the end of the current pass."""
        # Use the selected boundary or the natural video duration
        end_ms = self._playback_end_ms
        if end_ms is None:
            raise ValueError("Cannot determine the video's duration")

        # Completed playback has no remaining time, even when speed is zero
        remaining_ms = max(0.0, end_ms - self.status.time)
        if remaining_ms == 0:
            return 0.0

        # A stationary playback clock has no finite finish time
        if self.speed == 0:
            return float("inf")
        if self.speed < 0:
            raise ValueError("remaining_time requires nonnegative speed")

        return remaining_ms / (1000 * self.speed)

    def video_updater(self, mobj, dt):
        # Skip paused or closed playback without decoding another frame
        status = mobj.status
        if dt <= 0 or mobj.speed == 0 or status.closed:
            return
        if not isfinite(mobj.speed) or mobj.speed < 0:
            raise ValueError("speed must be finite and nonnegative")

        # Start audio (idempotent) when playback advances
        if (
            mobj.loop
            or mobj._playback_end_ms is None
            or status.time < mobj._playback_end_ms
        ):
            mobj._start_audio()

        next_time = status.time + 1000 * dt * mobj.speed
        end_ms = mobj._playback_end_ms
        seek = False
        finished = False

        # Wrap overshoot into the interval or select its final frame
        if end_ms is not None and next_time >= end_ms:
            if mobj.loop:
                mobj._loop_audio(next_time)
                duration = end_ms - mobj.start_ms
                next_time = mobj.start_ms + (next_time - mobj.start_ms) % duration
                seek = True
            else:
                next_time = end_ms
                finished = True

        # Keep the end exclusive while still decoding the final interval frame
        sample_time = np.nextafter(end_ms, -np.inf) if finished else next_time
        previous_frame = status.current_frame
        frame = mobj._read_frame(sample_time, seek=seek)
        status.time = next_time

        # Infer the natural end after decoding if duration metadata was missing
        if end_ms is None and status.exhausted and frame is not None:
            frame_duration = float(frame.duration * frame.time_base) * 1000
            if frame_duration <= 0:
                rate = status.stream.average_rate
                if not rate:
                    raise ValueError("Cannot determine the final frame duration")
                frame_duration = 1000 / float(rate)

            mobj._playback_end_ms = mobj._frame_time_ms(frame) + frame_duration

            # Apply the newly discovered end on the next update
            status.time = min(status.time, mobj._playback_end_ms)
            finished = not mobj.loop and status.time >= mobj._playback_end_ms

        # Convert pixels only when playback selects a different decoded frame
        if frame is not None and frame is not previous_frame:
            img = mobj._frame_image(frame)
            mobj.pixel_array = change_to_rgba_array(
                np.asarray(img), mobj.pixel_array_dtype
            )

        # Commit audio
        if finished:
            mobj._stop_audio(source_time=mobj._playback_end_ms / 1000)

            # Release the decoder after presenting the final frame
            if mobj.close_at_end:
                mobj.close()

    def seek(self, time: float) -> None:
        """Display an absolute source time in seconds, preserving pause state."""
        # Validate before interrupting the current playback interval
        target_ms = time * 1000
        if not isfinite(time) or target_ms < self.start_ms:
            raise ValueError("Seek time must be finite and at or after start_time")
        if self._playback_end_ms is not None and target_ms >= self._playback_end_ms:
            raise ValueError("Seek time must be before the playback end")

        # Commit audio before changing the source position
        self._stop_audio()
        was_paused = self.updating_suspended

        # Reopen a closed decoder and pause while preparing the requested frame
        if self.status.closed:
            self.restart()
            self.suspend_updating()

        # Decode the requested frame without advancing scene time
        frame = self._read_frame(target_ms, seek=True)
        if frame is None:
            raise ValueError("No video frame is available at the requested time")

        # Replace the image while preserving geometry and attached children
        img = self._frame_image(frame)
        self.pixel_array = change_to_rgba_array(np.asarray(img), self.pixel_array_dtype)
        self.orig_alpha_pixel_array = self.pixel_array[:, :, 3].copy()
        self.status.time = target_ms

        # Restore playback and start audio from the new source position
        if not was_paused:
            if self.updating_suspended:
                self.resume_updating()
            else:
                self._start_audio()

    def restart(self) -> None:
        """Reopen the source and resume playback from the selected start."""
        # Prepare a fresh decoder and the selected interval's initial frame
        fresh = VideoMobject(
            file_path=self.file_path,
            scene=self.scene,
            silent=True,
            imageops=self.imageops,
            speed=self.speed,
            loop=self.loop,
            start_time=self.start_ms / 1000,
            end_time=None if self.end_ms is None else self.end_ms / 1000,
            close_at_end=self.close_at_end,
        )

        # Close the previous decoder before adopting the reopened playback state
        self.close()
        # Keep the loaded audio when replacing the decoder's status
        fresh.status.audio = self.status.audio
        self.status = fresh.status
        self._stream_start = fresh._stream_start
        self.duration_ms = fresh.duration_ms
        self._playback_end_ms = fresh._playback_end_ms

        # Replace the frame without changing this object's displayed geometry
        self.pixel_array = fresh.pixel_array.copy()
        self.orig_alpha_pixel_array = fresh.orig_alpha_pixel_array.copy()

        # Attach this object's updater and resume playback
        self.add_updater(self.video_updater)
        self.resume_updating()

    def close(self):
        """Stop updating this object and close its shared video decoder."""
        # Release the container once, including when copies share this status
        self._stop_audio()
        self.remove_updater(self.video_updater)
        if not self.status.closed:
            self.status.container.close()
            self.status.closed = True

    def _start_audio(self, *, scene_time=None, source_time=None):
        # Leave an existing interval untouched and skip inactive playback
        if (
            self.silent
            or self.status.audio_recording
            or self.speed == 0
            or self.status.closed
            or self.updating_suspended
        ):
            return

        # Keep completed non-looping playback from opening another audio interval
        source_start = self.status.time / 1000 if source_time is None else source_time
        if (
            not self.loop
            and self._playback_end_ms is not None
            and source_start >= self._playback_end_ms / 1000
        ):
            return

        # Store both clocks in seconds to match Manim's audio timeline
        self.status.audio_scene_start = self.scene.time if scene_time is None else scene_time
        self.status.audio_source_start = source_start

        # Store speed because rounding makes calculating it from elapsed times unreliable
        self.status.audio_speed = self.speed
        self.status.audio_recording = True

    def _stop_audio(self, *, source_time=None):
        # Skip intervals that have already been committed
        if not self.status.audio_recording:
            return

        # Use the supplied source endpoint or the current video position
        source_end = self.status.time / 1000 if source_time is None else source_time
        source_duration = source_end - self.status.audio_source_start

        # Discard intervals where no playback time elapsed
        if source_duration <= 0:
            self.status.audio_recording = False
            return

        # Select the source samples consumed during this interval
        start_sample = round(self.status.audio_source_start * self.status.audio.frame_rate)
        end_sample = round(source_end * self.status.audio.frame_rate)
        segment = self.status.audio.get_sample_slice(start_sample, end_sample)

        # Use the speed recorded at the start of the interval
        playback_rate = max(1, round(segment.frame_rate * self.status.audio_speed))

        # Match playback speed, changing pitch along with it
        segment = segment._spawn(
            segment.raw_data,
            overrides={"frame_rate": playback_rate},
        ).set_frame_rate(self.status.audio.frame_rate)

        # Commit the audio at the scene time where playback began
        self.scene.renderer.file_writer.add_audio_segment(
            segment,
            time=self.status.audio_scene_start,
        )

        # Mark the interval committed so repeated stops cannot duplicate it
        self.status.audio_recording = False

    def _loop_audio(self, next_time: float):
        # Skip audio work when no interval is open
        if self.silent or not self.status.audio_recording:
            return

        # Match the video updater's source clock in milliseconds
        end_ms = self._playback_end_ms
        duration_ms = end_ms - self.start_ms

        # Handle every loop crossed by this update
        while next_time >= end_ms:
            # Place the next pass directly after the current audio interval
            next_audio_start = (
                self.status.audio_scene_start
                + (end_ms / 1000 - self.status.audio_source_start) / self.status.audio_speed
            )

            # Commit through the loop end and begin again at the loop start
            self._stop_audio(source_time=end_ms / 1000)
            self._start_audio(
                scene_time=next_audio_start,
                source_time=self.start_ms / 1000,
            )

            # Carry the remaining source time into the next pass
            next_time -= duration_ms

    def pause(self, recursive: bool = True):
        """Alias for suspend_updating"""
        return self.suspend_updating(recursive)

    def play(self, recursive: bool = True):
        """Alias for resume_updating"""
        return self.resume_updating(recursive)

    @override
    def suspend_updating(self, recursive: bool = True):
        result = super().suspend_updating(recursive)
        self._stop_audio()
        return result

    @override
    def resume_updating(self, recursive: bool = True):
        result = super().resume_updating(recursive)
        self._start_audio()
        return result


class UsageExample(Scene):
    def construct(self):
        video1 = (
            VideoMobject(
                file_path=("video1.mp4"),
                scene=self,
                speed=1.0,
            )
            .scale_to_fit_width(5)
            .to_corner(UL)
        )
        video2 = (
            VideoMobject(
                file_path=("video2.mkv"),
                scene=self,
                speed=3.0,
                loop=True,
                imageops=ImageOps.mirror,
            )
            .scale_to_fit_width(5)
            .to_corner(UR)
        )
        v1 = Group(video1, SurroundingRectangle(video1))
        v2 = Group(video2, SurroundingRectangle(video2))
        self.add(v1, v2)
        self.wait(2)
        self.play(v2.animate.shift(3 * DL), run_time=6)