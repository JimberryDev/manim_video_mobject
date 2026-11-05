
from collections.abc import Callable

import numpy as np
from manim import DL, Group
from manim.typing import Vector3D


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

    @staticmethod
    def reverse(tile_order: Callable[[Group], Group]) -> Callable[[Group], Group]:
        def order(tiles: Group) -> Group:
            # Apply the requested ordering, then return the tiles in reverse
            ordered = tile_order(tiles)
            return Group(*reversed(ordered))

        return order
