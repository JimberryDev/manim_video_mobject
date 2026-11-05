import warnings
from math import ceil

import numpy as np
from manim import ImageMobject
from manim.typing import Vector3D


class BlockMapper:
    """Measure a square block grid on an image or video."""

    def __init__(
        self,
        image: ImageMobject,
        *,
        reference_row: int | str | None = None,
        reference_column: int | str | None = None,
        reference_blocks: int,
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

        # Keep the wrapped image without taking ownership of its playback
        self.image = image

        # Measure the grid once from the image's current pixels
        reference_row, reference_column = self._resolve_reference(
            reference_row, reference_column
        )
        self._origin_px, self._block_size_px = self._measure_grid(
            reference_row, reference_column, reference_blocks
        )

        # Warn if the measured grid would contain unusually many blocks
        self._warn_if_grid_large()

    def _resolve_reference(
        self,
        row: int | str | None,
        column: int | str | None,
    ) -> tuple[int | None, int | None]:
        # Find visible pixels in the image
        visible = self.image.pixel_array[:, :, 3] > 0
        height, width = visible.shape

        assert isinstance(height, int) and isinstance(width, int)

        if row is not None:
            # Resolve the bottom alias or a Python-style row index
            if row == "bottom":
                rows = np.flatnonzero(visible.any(axis=1))
                if rows.size == 0:
                    raise ValueError("The image is fully transparent")
                row = int(rows[-1])
            assert isinstance(row, int)

            if row < 0:
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
                raise ValueError("The image is fully transparent")
            column = int(columns[0])
        assert isinstance(column, int)

        if column < 0:
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
        visible = self.image.pixel_array[:, :, 3] > 0

        if row is not None:
            # Measure the horizontal span and anchor below its left edge
            positions = np.flatnonzero(visible[row, :])
            origin = (int(positions[0]), row + 1)
        else:
            assert column is not None
            # Measure the vertical span and anchor below its left edge
            positions = np.flatnonzero(visible[:, column])
            origin = (column, int(positions[-1]) + 1)

        # Include both endpoint pixels and divide the span into square blocks
        span = int(positions[-1] - positions[0] + 1)
        block_size = round(span / blocks)
        if block_size < 1:
            raise ValueError("The measured block size is less than one pixel")

        return origin, block_size

    def _warn_if_grid_large(self):
        # Count grid cells, including transparent and partial cells
        height, width = self.image.pixel_array.shape[:2]
        block_size = self._block_size_px
        offset_x = self._origin_px[0] % block_size
        offset_y = self._origin_px[1] % block_size
        columns = ceil((width - offset_x) / block_size) + int(offset_x != 0)
        rows = ceil((height - offset_y) / block_size) + int(offset_y != 0)
        total_blocks = columns * rows

        # Warn when the grid exceeds 121 blocks
        if total_blocks > 121:
            warnings.warn(
                f"The measured grid has {total_blocks} blocks "
                f"({columns} columns x {rows} rows)",
                stacklevel=3,
            )

    def pixel_to_point(self, x: float, y: float) -> Vector3D:
        # Map source pixel boundaries onto the wrapped image's current geometry
        height, width = self.image.pixel_array.shape[:2]
        upper_left, upper_right, lower_left, _ = self.image.points

        return (
            upper_left
            + (upper_right - upper_left) * (x / width)
            + (lower_left - upper_left) * (y / height)
        )

    def get_origin_px(self) -> tuple[int, int]:
        # Return the stored bottom-left grid intersection as (x, y)
        return self._origin_px

    def get_block_size_px(self) -> int:
        # Return the stored square block size in pixels
        return self._block_size_px

    def get_origin_manim(self) -> Vector3D:
        # Map the stored pixel origin onto the current image geometry
        return self.pixel_to_point(*self.get_origin_px())

    def get_block_size_manim(self) -> float:
        # Measure one block along the current image's horizontal edge
        x, y = self.get_origin_px()
        start = self.pixel_to_point(x, y)
        end = self.pixel_to_point(x + self.get_block_size_px(), y)
        return float(np.linalg.norm(end - start))

    def get_block_center(self, x: int, y: int) -> Vector3D:
        # Find the block center, with block Y increasing upward
        origin_x, origin_y = self.get_origin_px()
        size = self.get_block_size_px()

        # Map the center onto the current image geometry
        return self.pixel_to_point(
            origin_x + (x + 0.5) * size,
            origin_y - (y + 0.5) * size,
        )

    def get_block_bounds_px(
        self, x: int, y: int
    ) -> tuple[int, int, int, int]:
        # Find the block boundaries, with block Y increasing upward
        origin_x, origin_y = self.get_origin_px()
        size = self.get_block_size_px()

        left = origin_x + x * size
        right = left + size
        bottom = origin_y - y * size
        top = bottom - size

        # Return absolute pixel boundaries without clipping or creating an object
        return left, top, right, bottom
