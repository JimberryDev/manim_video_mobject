"""Render with Manim to check playback state and listen for audio sync and seams."""

from pathlib import Path

from manim import UP, Scene, Text, config

from manim_video import VideoMobject


class AudioTest(Scene):
    def construct(self):
        # Locate the shared source and allow ten frames for endpoint checks
        path = Path(__file__).resolve().parent / "res" / "Hilbert floor.mp4"
        frame_margin = 10 / config.frame_rate

        # Prepare a short clip whose decoder stays open at its endpoint
        video = VideoMobject(path, scene=self, end_time=40, close_at_end=False)
        video.scale_to_fit_width(10)
        self.add(video)
        assert video.status.audio is not None, "This scene needs source audio"

        # Start playback automatically and listen for normal audio
        self._show_stage("Normal playback")
        self.wait(5)
        assert video.status.audio_interval is not None

        # Repeated play calls must preserve the already open audio interval
        interval = video.status.audio_interval
        video.play()
        video.play()
        assert video.status.audio_interval is interval

        # Pause twice and confirm that the source clock stays still
        self._show_stage("Paused: expect silence")
        video.pause()
        video.pause()
        paused_time = video.status.time
        self.wait(2.5)
        assert video.status.time == paused_time
        assert video.status.audio_interval is None

        # Resume and listen for audio continuing from the paused position
        self._show_stage("Resume at normal speed")
        video.play()
        self.wait(5)

        # Change speeds during playback and listen for matching pitch changes
        self._show_stage("Double speed")
        video.speed = 2
        self.wait(5)
        self._show_stage("Half speed")
        video.speed = 0.5
        self.wait(5)

        # Zero speed must commit audio and hold the source clock
        self._show_stage("Zero speed: expect silence")
        video.speed = 0
        stationary_time = video.status.time
        self.wait(2.5)
        assert video.status.time == stationary_time
        assert video.status.audio_interval is None

        # Seek during playback and listen for audio jumping to the same position
        self._show_stage("Seek to source second 10")
        video.speed = 1
        video.seek(10)
        self.wait(3.75)

        # Seeking while paused must not resume either video or audio
        self._show_stage("Paused seek: expect silence")
        video.pause()
        video.seek(5)
        self.wait(2.5)
        assert video.updating_suspended
        assert video.status.time == 5000
        assert video.status.audio_interval is None

        # Restart should reuse the loaded audio and resume from the selected start
        self._show_stage("Restart from the beginning")
        audio = video.status.audio
        video.restart()
        assert video.status.audio is audio
        self.wait(3.75)

        # Reach the endpoint while keeping the decoder open
        self._show_stage("Finish clip, then hold silently")
        video.seek(37.5)
        self.wait(video.remaining_time() + frame_margin)
        assert not video.status.closed
        assert video.status.audio_interval is None

        # Play and speed changes must not reopen completed audio
        video.play()
        video.speed = 2
        self.wait(2.5)
        assert video.status.audio_interval is None

        # Seek backward after completion and verify that playback can start again
        self._show_stage("Seek backward after completion")
        video.speed = 1
        video.seek(5)
        self.wait(2.5)
        assert video.status.audio_interval is not None

        # Remove the clip and explicitly commit audio and close its decoder
        self.remove(video)
        video.close()
        video.close()
        assert video.status.closed
        assert video.status.audio_interval is None

        # Cross several loop boundaries and listen for missing or duplicated audio
        self._show_stage("Loop source seconds 0 to 3.75")
        loop = VideoMobject(path, scene=self, end_time=3.75, loop=True)
        loop.scale_to_fit_width(10)
        self.add(loop)
        self.wait(12.5)
        assert 0 <= loop.status.time < 3750

        # Repeat loop boundaries at a faster playback speed
        self._show_stage("Loop at double speed")
        loop.speed = 2
        self.wait(7.5)
        assert 0 <= loop.status.time < 3750
        self.remove(loop)
        loop.close()

        # Check automatic decoder closure and seeking after a closed endpoint
        self._show_stage("Automatic close, then seek and resume")
        ending = VideoMobject(path, scene=self, end_time=3.75)
        ending.scale_to_fit_width(10)
        self.add(ending)
        self.wait(ending.remaining_time() + frame_margin)
        assert ending.status.closed
        assert ending.status.audio_interval is None
        ending.seek(0)
        self.wait(1.25)
        assert not ending.status.closed
        assert ending.status.audio_interval is not None
        self.remove(ending)
        ending.close()

        # Leave a silent tail to make audio leaking past cleanup easy to hear
        self._show_stage("Closed: expect silence")
        self.wait(2.5)
        print("Playback state checks passed; listen to the render for audio sync and seams")

    def _show_stage(self, message: str):
        # Replace the stage label without advancing the scene clock
        if hasattr(self, "stage_label"):
            self.remove(self.stage_label)
        self.stage_label = Text(message, font_size=28).to_edge(UP)
        self.add(self.stage_label)
