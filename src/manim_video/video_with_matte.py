from pathlib import Path
from typing import Self

from manim import Scene
from PIL import Image

from .video_mobject import VideoMobject


class VideoWithMatte(VideoMobject):
    """Use one video's colors and another video's alpha channel."""

    def __init__(
        self,
        file_path: str | Path,
        matte_path: str | Path,
        scene: Scene,
        **kwargs,
    ):
        # Store the matte source for restarting and cloning
        self.matte_path = Path(matte_path)

        # Prepare the matte before the main constructor converts its first frame
        self.matte = VideoMobject(
            self.matte_path,
            scene=scene,
            silent=True,
            start_time=kwargs.get("start_time", 0),
            close_at_end=False,
            ignore_transparent=False,
        )

        # Let the main video's clock drive the matte decoder
        self.matte.remove_updater(self.matte.video_updater)

        try:
            # Initialize this object as the main video
            # Image operations apply to the combined image, not the matte alone
            super().__init__(file_path, scene=scene, **kwargs)

            # Reject a known matte endpoint before the selected main endpoint
            main_end = self._playback_end_ms
            matte_end = self.matte._playback_end_ms
            if main_end is not None and matte_end is not None and matte_end < main_end:
                raise ValueError("The matte ends before the selected video interval")

        except Exception:
            # Release both decoders if setup or validation fails
            if hasattr(self, "status"):
                VideoMobject.close(self)
            self.matte.close()
            raise

    def _get_matte_frame(self, time_ms: float):
        # Reopen the matte if main playback previously closed both decoders
        reopened = self.matte.status.closed
        if reopened:
            self.matte.restart()
            self.matte.remove_updater(self.matte.video_updater)

        # Reject positions beyond a known matte endpoint
        end_ms = self.matte._playback_end_ms
        if end_ms is not None and time_ms >= end_ms:
            raise ValueError("The requested time is beyond the matte's end")

        # Seek when moving backward, including main-video loops and seeks
        seek = reopened or time_ms < self.matte.status.time
        frame = self.matte._read_frame(time_ms, seek=seek)
        if frame is None:
            raise ValueError("No matte frame is available at the requested time")

        # Remember the position so the next request can detect backward movement
        self.matte.status.time = time_ms
        return frame

    def _frame_image(self, frame):
        # Convert the main frame without applying image operations yet
        image = Image.fromarray(frame.to_ndarray(format="rgba"))

        # Select the matte at the main frame's source timestamp
        time_ms = self._frame_time_ms(frame)
        matte_frame = self._get_matte_frame(time_ms)
        matte_image = Image.fromarray(matte_frame.to_ndarray(format="rgba"))

        # Require matching dimensions before transferring the alpha channel
        if image.size != matte_image.size:
            raise ValueError("Main video and matte must have matching dimensions")

        # Keep the main colors and replace only their transparency
        image.putalpha(matte_image.getchannel("A"))

        # Apply image operations to the combined result
        if self.imageops is not None:
            image = self.imageops(image)

        return image

    def clone(self) -> Self:
        # Clone the main video with its own decoder and playback state
        cloned = super().clone()

        try:
            # Replace the shared matte with an independent decoder
            # Do not close the old reference because the original still uses it
            cloned.matte = self.matte.clone()
            cloned.matte.remove_updater(cloned.matte.video_updater)

        except Exception:
            # Close only the new main decoder if cloning the matte fails
            VideoMobject.close(cloned)
            raise

        return cloned

    def close(self):
        # Close the matte decoder
        try:
            self.matte.close()
        finally:
            # Commit main audio and close its decoder
            super().close()
