import numpy as np
from manim import ImageMobject
from manim.typing import Vector3D

from .video_mobject import VideoMobject


class McVideo(VideoMobject):
    def __init__(
        self,
        filename=None,
        *,
        reference_row: int | str | None = None,
        reference_column: int | str | None = None,
        reference_blocks: int,
        imageops=None,
        speed=1.0,
        loop=False,
        start_time=0.0,
        end_time=None,
        close_at_end=True,
        **kwargs,
    ):
        # Require exactly one reference row or column
        if (reference_row is None) == (reference_column is None):
            raise ValueError("Specify exactly one of reference_row or reference_column")

        # Require a positive whole number of blocks
        if type(reference_blocks) is not int or reference_blocks <= 0:
            raise ValueError("reference_blocks must be a positive integer")

        # Accept pixel indices and the supported edge aliases
        if reference_row is not None:
            if type(reference_row) is not int and reference_row != "bottom":
                raise ValueError("reference_row must be an integer or 'bottom'")
        else:
            if type(reference_column) is not int and reference_column != "left":
                raise ValueError("reference_column must be an integer or 'left'")

        # Load the initial frame with the caller's image operations
        super().__init__(
            file_path=filename,
            imageops=imageops,
            speed=speed,
            loop=loop,
            start_time=start_time,
            end_time=end_time,
            close_at_end=close_at_end,
            **kwargs,
        )

        # Enable visible bounds after Manim finishes sizing the full image
        self._use_visible_bounds = True

        # Resolve the reference line and measure the grid in the initial frame
        try:
            reference_row, reference_column = self._resolve_reference(
                reference_row, reference_column
            )
            self._origin_px, self._tile_size_px = self._measure_grid(
                reference_row, reference_column, reference_blocks
            )
        except Exception:
            # Release the decoder if reference selection or grid measurement fails
            self.close()
            raise

    def _resolve_reference(
        self,
        row: int | str | None,
        column: int | str | None,
    ) -> tuple[int | None, int | None]:
        # Find visible pixels in the initial frame
        visible = self.pixel_array[:, :, 3] > 0
        height, width = visible.shape

        if row is not None:
            # Resolve the bottom alias or a Python-style row index
            if row == "bottom":
                rows = np.flatnonzero(visible.any(axis=1))
                if rows.size == 0:
                    raise ValueError("The initial frame is fully transparent")
                row = int(rows[-1])
            elif row < 0:
                row += height

            # Require a row inside the frame with some visible pixels
            if not 0 <= row < height:
                raise ValueError("reference_row is outside the frame")
            if not visible[row, :].any():
                raise ValueError("reference_row is fully transparent")

            return row, None

        # Resolve the left alias or a Python-style column index
        if column == "left":
            columns = np.flatnonzero(visible.any(axis=0))
            if columns.size == 0:
                raise ValueError("The initial frame is fully transparent")
            column = int(columns[0])
        elif column < 0:
            column += width

        # Require a column inside the frame with some visible pixels
        if not 0 <= column < width:
            raise ValueError("reference_column is outside the frame")
        if not visible[:, column].any():
            raise ValueError("reference_column is fully transparent")

        return None, column

    def _measure_grid(
        self,
        row: int | None,
        column: int | None,
        blocks: int,
    ) -> tuple[tuple[int, int], int]:
        # Find visible pixels along the resolved reference line
        visible = self.pixel_array[:, :, 3] > 0

        if row is not None:
            # Measure the horizontal span and anchor below its left edge
            positions = np.flatnonzero(visible[row, :])
            origin = (int(positions[0]), row + 1)
        else:
            # Measure the vertical span and anchor below its left edge
            positions = np.flatnonzero(visible[:, column])
            origin = (column, int(positions[-1]) + 1)

        # Include both endpoint pixels and divide the span into square blocks
        span = int(positions[-1] - positions[0] + 1)
        tile_size = round(span / blocks)
        if tile_size < 1:
            raise ValueError("The measured tile size is less than one pixel")

        return origin, tile_size

    # Parent groups use raw child points for some bounds calculations
    # These visible bounds apply directly to McVideo, not to those group calculations
    def get_points_defining_boundary(self):
        # Use normal bounds while Manim prepares the image geometry
        if not getattr(self, "_use_visible_bounds", False):
            return super().get_points_defining_boundary()

        # Find the current frame's visible rows and columns
        visible = self.pixel_array[:, :, 3] > 0
        rows = np.flatnonzero(visible.any(axis=1))
        columns = np.flatnonzero(visible.any(axis=0))

        # Anchor a fully transparent frame at the full image's center
        if rows.size == 0:
            boundary = self.points.mean(axis=0, keepdims=True)
        else:
            # Include the outer boundaries of the visible pixels
            left, right = columns[0], columns[-1] + 1
            top, bottom = rows[0], rows[-1] + 1

            # Map the visible rectangle onto the displayed image
            boundary = np.array(
                [
                    self.pixel_to_point(left, top),
                    self.pixel_to_point(right, top),
                    self.pixel_to_point(left, bottom),
                    self.pixel_to_point(right, bottom),
                ]
            )

        # Include attached labels and other child objects
        return np.concatenate(
            [
                boundary,
                *(child.get_points_defining_boundary() for child in self.submobjects),
            ]
        )

    def reduce_across_dimension(self, reduce_func, dim):
        # Preserve normal sizing while Manim prepares the image geometry
        if not getattr(self, "_use_visible_bounds", False):
            return super().reduce_across_dimension(reduce_func, dim)

        # Measure the requested axis using the same visible bounds
        points = self.get_points_defining_boundary()
        return reduce_func(points[:, dim])

    def pixel_to_point(self, x: float, y: float) -> Vector3D:
        # Map pixel boundaries onto the current image geometry
        height, width = self.pixel_array.shape[:2]
        upper_left, upper_right, lower_left, _ = self.points

        return (
            upper_left
            + (upper_right - upper_left) * (x / width)
            + (lower_left - upper_left) * (y / height)
        )

    def get_origin_px(self) -> tuple[int, int]:
        # Return the stored bottom-left grid intersection as (x, y)
        return self._origin_px

    def get_tile_size_px(self) -> int:
        # Return the stored square block size in pixels
        return self._tile_size_px

    def get_origin_manim(self) -> Vector3D:
        # Map the stored pixel origin onto the current image geometry
        return self.pixel_to_point(*self.get_origin_px())

    def get_tile_size_manim(self) -> float:
        # Measure one block along the current image's horizontal edge
        x, y = self.get_origin_px()
        start = self.pixel_to_point(x, y)
        end = self.pixel_to_point(x + self.get_tile_size_px(), y)
        return float(np.linalg.norm(end - start))

    def get_block_center(self, x: int, y: int) -> Vector3D:
        # Find the block center, with block Y increasing upward
        origin_x, origin_y = self.get_origin_px()
        size = self.get_tile_size_px()

        # Map the center onto the current image geometry
        return self.pixel_to_point(
            origin_x + (x + 0.5) * size,
            origin_y - (y + 0.5) * size,
        )

    def get_block(self, x, y) -> ImageMobject:
        # Find the block boundaries, with block Y increasing upward
        origin_x, origin_y = self.get_origin_px()
        size = self.get_tile_size_px()

        left = origin_x + x * size
        right = left + size
        bottom = origin_y - y * size
        top = bottom - size

        # Copy the current frame's block pixels
        pixels = self.pixel_array[top:bottom, left:right].copy()
        return ImageMobject(pixels)

    def crop(self, left: int, right: int, top: int, bottom: int) -> VideoMobject:
        # Keep existing image operations before cropping each frame
        previous_imageops = self.imageops

        def cropped_imageops(image):
            # chain the previous imageops and this one: newops(prevops())
            if previous_imageops is not None:
                image = previous_imageops(image)
            return image.crop((
                left,
                top,
                image.width - right,
                image.height - bottom,
            ))

        # Apply the crop whenever the decoder presents a frame
        self.imageops = cropped_imageops
        return self
