from typing import override

from manim import Scene


class Fix(Scene):
    @override
    def play_internal(self, skip_rendering: bool = False) -> None:
        """
        This method is used to prep the animations for rendering,
        apply the arguments and parameters required to them,
        render them, and write them to the video file.

        Parameters
        ----------
        skip_rendering
            Whether the rendering should be skipped, by default False
        """
        assert self.animations is not None
        self.duration = self.get_run_time(self.animations)
        self.time_progression = self._get_animation_time_progression(
            self.animations,
            self.duration,
        )
        for t in self.time_progression:
            self.update_to_time(t)
            if not skip_rendering and not self.skip_animation_preview:
                self.renderer.render(self, t, self.moving_mobjects)
            if self.stop_condition is not None and self.stop_condition():
                self.time_progression.close()
                break

        # This is the fix:
        # Deliver the remaining interval when playback reaches its normal end
        if self.stop_condition is None and self.last_t < self.duration:
            self.update_to_time(self.duration)

        for animation in self.animations:
            animation.finish()
            animation.clean_up_from_scene(self)
        if not self.renderer.skip_animations:
            self.update_mobjects(0)
        # TODO: The OpenGLRenderer does not have the property static.image.
        self.renderer.static_image = None  # type: ignore[union-attr]
        # Closing the progress bar at the end of the play.
        self.time_progression.close()
