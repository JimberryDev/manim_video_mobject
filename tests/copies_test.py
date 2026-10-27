from pathlib import Path

from manim import LEFT, RIGHT, Scene

from manim_video import VideoMobject


class CopiesTest(Scene):
    def construct(self):
        # Locate the source video relative to this test file
        path = Path(__file__).resolve().parent / "res" / "Hilbert floor.mp4"
        

        # Prepare two independent decoders at the same source position
        left = VideoMobject(path, scene=self, silent=False)
        right = VideoMobject(path, scene=self, silent=False)
        left.scale_to_fit_width(5).shift(LEFT * 3)
        right.scale_to_fit_width(5).shift(RIGHT * 3)

        # Measure source advancement during two scene seconds
        self.add(left, right)
        self.wait(2, frozen_frame=False)
        print(f"Independent: {left.status.time / 1000:.3f} {right.status.time / 1000:.3f}")

        # Remove and close both independent decoders
        self.remove(left, right)
        left.close()
        right.close()

        # Prepare two displayed copies sharing one decoder and clock
        left = VideoMobject(path, scene=self, silent=False)
        left.scale_to_fit_width(5).shift(LEFT * 3)
        right = left.copy().shift(RIGHT * 6)

        # Repeat the same wait with the shared playback clock
        self.add(left, right)
        self.wait(2, frozen_frame=False)
        print(f"Copied: {left.status.time / 1000:.3f} {right.status.time / 1000:.3f}")
        print("Shared status:", left.status is right.status)

        # Remove both views before closing their shared decoder
        self.remove(left, right)
        left.close()
        right.close()
