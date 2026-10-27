from collections.abc import Callable
from itertools import pairwise

import numpy as np
from manim import DL, Animation, Camera, Group, ImageMobject, LaggedStart, Vector3D

from .block_mapper import BlockMapper


def grid_edges(size: int, tile_size: int, origin: int) -> list[int]:
    # Extend the grid through the origin and clip at the image edges
    offset = origin % tile_size
    return [0, *range(offset or tile_size, size, tile_size), size]


def make_image_tiles(
    image: ImageMobject,
    tile_width: int = 128,
    tile_height: int = 128,
    grid_origin: tuple[int, int] = (0, 0),
    camera: Camera | None = None,
) -> Group:
    """Crop source pixels, optionally snapping geometry to a fixed Cairo camera."""
    # Require whole pixels for tile sizes and grid alignment
    if any(
        not isinstance(value, int) or value <= 0 for value in (tile_width, tile_height)
    ):
        raise ValueError("Tile dimensions must be positive integers")

    origin_x, origin_y = grid_origin
    if any(not isinstance(value, int) for value in grid_origin):
        raise ValueError("Grid origin coordinates must be integers")

    # Calculate crop boundaries from the source image
    pixels = image.pixel_array
    height, width = pixels.shape[:2]
    x_edges = grid_edges(width, tile_width, origin_x)
    y_edges = grid_edges(height, tile_height, origin_y)

    # Use the image's actual corners so rotated images keep their orientation
    upper_left, upper_right, lower_left, _ = image.get_points()
    across = upper_right - upper_left
    downward = lower_left - upper_left

    # Calculate each shared grid intersection once before assigning tile corners
    grid_points = np.array(
        [
            upper_left + across * (x / width) + downward * (y / height)
            for y in y_edges
            for x in x_edges
        ]
    )

    # Snap final positions to output pixels without changing the source crops
    if camera is not None:
        screen_points = camera.points_to_subpixel_coords(image, grid_points)
        adjustment = np.rint(screen_points) - screen_points
        grid_points[:, 0] += adjustment[:, 0] * camera.frame_width / camera.pixel_width
        grid_points[:, 1] -= (
            adjustment[:, 1] * camera.frame_height / camera.pixel_height
        )

    # Share the same corner coordinates between neighboring tiles
    grid_points = grid_points.reshape(len(y_edges), len(x_edges), 3)

    # Build independent crops at their final positions
    tiles = Group()
    for row, (y0, y1) in enumerate(pairwise(y_edges)):
        for column, (x0, x1) in enumerate(pairwise(x_edges)):
            tile = ImageMobject(pixels[y0:y1, x0:x1].copy())
            tile.set_z_index(image.z_index)

            # Match Manim's image corner order when mapping pixels to the scene
            tile.set_points(
                np.array(
                    [
                        grid_points[row, column],
                        grid_points[row, column + 1],
                        grid_points[row + 1, column],
                        grid_points[row + 1, column + 1],
                    ]
                )
            )
            tiles.add(tile)

    return tiles


class TileOrder:
    """Reusable tile ordering callbacks for TileAnimation."""

    @staticmethod
    def distance_from_bottom_left(tiles: Group) -> Group:
        """Order tile centers by distance, then angle from 90 degrees to 0."""
        # Measure positions from the bottom-left of the tile group's bounds
        bottom_left = tiles.get_corner(DL)

        def sort_key(tile):
            offset = tile.get_center() - bottom_left
            distance = np.linalg.norm(offset[:2])
            angle = np.arctan2(offset[1], offset[0])
            return distance, -angle

        # Return the same tiles in their animation order
        return Group(*sorted(tiles, key=sort_key))

    @staticmethod
    def distance_from_point(point: Vector3D) -> Callable[[Group], Group]:
        # Store the reference position when creating the callback
        origin = np.array(point, dtype=float).copy()

        def order(tiles: Group) -> Group:
            # Measure distance and angle from the supplied scene position
            def sort_key(tile):
                offset = tile.get_center() - origin
                distance = np.linalg.norm(offset[:2])
                angle = np.arctan2(offset[1], offset[0])
                return distance, -angle

            # Return the same tiles ordered nearest first
            return Group(*sorted(tiles, key=sort_key))

        return order


class TileAnimation(LaggedStart):
    """Apply staggered animations to crops of an image.

    Parameters
    ----------
    image
        Source image whose current pixels are copied into tiles
        A BlockMapper supplies its wrapped image, block size, and grid origin
        When a mapper is supplied, its grid overrides tile_width,
        tile_height, and grid_origin
    animation_factory
        Function receiving each tile and returning its animation
    tile_width, tile_height
        Tile dimensions in source pixels
        Tiles at the image edges may be smaller
    grid_origin
        Grid intersection in source pixels, measured from the top-left
        Positive X points right and positive Y points down
    lag_ratio
        Delay between tile starts relative to an individual animation's duration
    run_time
        Total duration of the staggered animation in seconds
    restore_original
        Replace the tiles with the original image after the animation
        Transitions should end with tiles assembled to avoid a visible jump
    ignore_empty
        Skip tiles where at least 90 percent of pixels have zero alpha
    tile_order
        Optional function receiving the filtered tile Group and returning
        a Group in the desired animation order
        None keeps top-to-bottom, left-to-right order
    camera
        Optional fixed Cairo camera used to snap assembled tile boundaries
        to output pixels, slightly adjusting geometry while keeping crops intact
        Camera motion or later scaling can invalidate the alignment
    **kwargs
        Additional arguments passed to LaggedStart
    """

    def __init__(
        self,
        image: ImageMobject | BlockMapper,
        animation_factory: Callable[[ImageMobject], Animation],
        tile_width: int = 128,
        tile_height: int = 128,
        grid_origin: tuple[int, int] = (0, 0),
        lag_ratio: float = 0.08,
        run_time: float = 2.0,
        restore_original: bool = True,
        ignore_empty: bool = True,
        tile_order: Callable[[Group], Group] | None = None,
        camera: Camera | None = None,
        **kwargs,
    ):
        # Use the mapper's image and measured block grid when supplied
        if isinstance(image, BlockMapper):
            tile_width = tile_height = image.get_block_size_px()
            grid_origin = image.get_origin_px()
            image = image.image

        # Prepare image crops before constructing their transitions
        self.image = image
        self.restore_original = restore_original
        self.tiles = make_image_tiles(
            image, tile_width, tile_height, grid_origin, camera=camera
        )

        # Keep tiles with more than ten percent visible pixels
        if ignore_empty:
            self.tiles = Group(
                *(
                    tile
                    for tile in self.tiles
                    if np.count_nonzero(tile.pixel_array[:, :, 3] > 0)
                    / tile.pixel_array[:, :, 3].size
                    > 0.1
                )
            )

        # Apply the caller's tile order before building transitions
        if tile_order is not None:
            self.tiles = tile_order(self.tiles)

        # Let Manim stagger the caller's transition for each tile
        super().__init__(
            *(animation_factory(tile) for tile in self.tiles),
            group=self.tiles,
            lag_ratio=lag_ratio,
            run_time=run_time,
            **kwargs,
        )

    def _setup_scene(self, scene):
        # Hide the original while its tiles animate
        scene.remove(self.image)
        super()._setup_scene(scene)
        scene.add(self.tiles)

    def clean_up_from_scene(self, scene):
        # Finish each transition before optionally restoring the whole image
        super().clean_up_from_scene(scene)
        if self.restore_original:
            scene.remove(self.tiles, *self.tiles)
            scene.add(self.image)
