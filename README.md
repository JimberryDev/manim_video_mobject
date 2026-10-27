# manim-video

Video playback and tiled image animations for Manim Community.

Requires Python 3.14+, Manim 0.22+, and FFmpeg available on PATH for audio loading.

## Setup

From this repository:

```sh
uv sync
```

## VideoMobject

```python
from manim import FadeIn, FadeOut, Scene
from manim_video import VideoMobject


class Example(Scene):
    def construct(self):
        # Load a source interval, including its audio
        video = VideoMobject(
            "clip.mp4",
            scene=self,
            start_time=0,
            end_time=5,
        )
        video.scale_to_fit_width(10)

        # Reveal, play to the endpoint, then remove and close
        self.play(FadeIn(video))
        self.wait(video.remaining_time())
        self.play(FadeOut(video))
        video.close()
```

Video transparency is preserved. Use `silent=True` to skip audio,
`loop=True` to repeat the selected interval, and `speed` to change playback
speed. Speed changes also change audio pitch.

`video.pause()` suspends playback.

`video.play()` resumes playback.

`video.seek(seconds)` changes the source position.

`video.restart()` returns to the selected start.


Always call `video.close()` after removing it from the scene, and close
remaining videos before the scene ends. Manim does not notify objects
when they are removed, so you have to do it yourself.

Copies share the decoder, playback clock, and audio state. Construct
another `VideoMobject` for independent playback.

## Other helpers

- `BlockMapper`: measures a square block grid on an image or video. Provides block sizes, origins, centers, and pixel bounds. Measurements are captured at construction; scene coordinates follow the wrapped image's current geometry.



- `TileAnimation` and `TileOrder`: staggered animations over image tiles
- `Fix`: scene subclass that delivers the final playback update

## Render checks

The scenes in `tests/` use `tests/res/Hilbert floor.mp4`.

```sh
uv run manim --disable_caching tests/copies_test.py CopiesTest
uv run manim --disable_caching tests/audio_test.py AudioTest
```

The audio scene checks playback state; listen to its output to assess
audio synchronization and loop seams.
