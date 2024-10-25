# annotations.py

import csv
import dataclasses
import os
import ast
import itertools
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


@dataclasses.dataclass
class AnnotatedPosition:
    frame: int
    x: int
    y: int
    # Angle is annotated in vector notation in the same coordinate
    # system as the x, y coordinates.
    u: float = 0.0
    v: float = 0.0


class Annotations:
    @staticmethod
    def get_annotation_index_for_frame(
        list_of_positions: List[AnnotatedPosition], frame: int
    ) -> Optional[int]:
        index = [idx for idx, position in enumerate(
            list_of_positions) if position.frame == frame]
        assert len(index) < 2
        if len(index) == 0:
            return None
        return index[0]

    def __init__(self):
        self.raw_thorax_positions: List[AnnotatedPosition] = []
        self.waggle_starts: List[AnnotatedPosition] = []

    def update_thorax_position(self, frame: int, x: int, y: int):
        existing_index = Annotations.get_annotation_index_for_frame(
            self.raw_thorax_positions, frame
        )
        if existing_index is not None:
            self.raw_thorax_positions[existing_index].x = x
            self.raw_thorax_positions[existing_index].y = y
        else:
            self.raw_thorax_positions.append(AnnotatedPosition(frame, x, y))

    def update_waggle_start(
        self, frame: int, x: int, y: int, u: float = np.nan, v: float = np.nan
    ):
        existing_index = Annotations.get_annotation_index_for_frame(
            self.waggle_starts, frame
        )
        if existing_index is not None:
            self.waggle_starts[existing_index].x = x
            self.waggle_starts[existing_index].y = y
            if not pd.isnull(u):
                self.waggle_starts[existing_index].u = u
                self.waggle_starts[existing_index].v = v
        else:
            self.waggle_starts.append(AnnotatedPosition(frame, x, y, u, v))

    def update_waggle_direction(self, frame: int, to_x: int, to_y: int):
        existing_index = Annotations.get_annotation_index_for_frame(
            self.waggle_starts, frame
        )
        if existing_index is None:
            return
        direction = np.array(
            [
                to_x - self.waggle_starts[existing_index].x,
                to_y - self.waggle_starts[existing_index].y,
            ],
            dtype=np.float64,
        )
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0.0:
            direction /= direction_norm
            self.waggle_starts[existing_index].u = direction[0]
            self.waggle_starts[existing_index].v = direction[1]
        else:
            self.waggle_starts[existing_index].u = 0.0
            self.waggle_starts[existing_index].v = 0.0

    def clear(self):
        self.raw_thorax_positions.clear()
        self.waggle_starts.clear()

    def calculate_min_max_thorax_distance_to_actuator(
        self, actuator: Tuple[int, int]
    ) -> Optional[Tuple[float, float]]:
        if len(self.raw_thorax_positions) == 0 or not actuator:
            return None
        actuator_array = np.array(actuator)
        thorax_xy = np.array([(p.x, p.y) for p in self.raw_thorax_positions])
        distances = np.linalg.norm(thorax_xy - actuator_array, axis=1)
        return np.around(distances.min(), decimals=2), np.around(
            distances.max(), decimals=2
        )

    def get_maximum_annotated_frame_index(self) -> Optional[int]:
        try:
            frame = max(
                [
                    max([a.frame for a in annotation_list])
                    for annotation_list in (
                        self.waggle_starts,
                        self.raw_thorax_positions,
                    )
                    if annotation_list
                ]
            )
        except ValueError:
            frame = None
        return frame

    def is_empty(self) -> bool:
        return self.get_maximum_annotated_frame_index() is None

    @staticmethod
    def load(filepath: str, on_error: str = "print") -> List['Annotations']:
        def parse_string_list(values):
            if isinstance(values, list):
                values = map(ast.literal_eval, values)
                values = list(itertools.chain(*values))
            else:
                values = ast.literal_eval(values)
            return values

        try:
            all_annotations = []

            annotations_df = pd.read_csv(
                get_output_filename(filepath),
                decimal=".",
                **get_csv_writer_options(),
            )

            def to_leaf_name(path):
                return os.path.basename(path)

            annotations_df = annotations_df[
                annotations_df["video_name"].apply(to_leaf_name)
                == to_leaf_name(filepath)
            ]

            # Each row constitutes a different set of annotations.
            for row_index in range(annotations_df.shape[0]):
                row_df = annotations_df.iloc[row_index: (row_index + 1)]
                annotations = Annotations()

                for idx, (all_xy, all_frames) in enumerate(
                    row_df[
                        ["thorax_positions", "thorax_frames"]
                    ].itertuples(index=False)
                ):
                    # We only load one row per annotation-session here.
                    assert idx == 0

                    all_xy = parse_string_list(all_xy)
                    all_frames = parse_string_list(all_frames)

                    for (xy, frame) in zip(all_xy, all_frames):
                        annotations.update_thorax_position(frame, xy[0], xy[1])

                for all_xy, all_frames, all_uv in row_df[
                    [
                        "waggle_start_positions",
                        "waggle_start_frames",
                        "waggle_directions",
                    ]
                ].itertuples(index=False):
                    all_xy = parse_string_list(all_xy)
                    all_uv = parse_string_list(all_uv)
                    all_frames = parse_string_list(all_frames)

                    for (xy, frame, uv) in zip(all_xy, all_frames, all_uv): 
                        annotations.update_waggle_start(
                            frame, xy[0], xy[1], uv[0], uv[1]
                        )

                all_annotations.append(annotations)

            return all_annotations
        except Exception as e:
            if on_error == "print":
                print(
                    "Could not read old annotations. Continuing normally. "
                    f"Error: {repr(e)}"
                )
            elif on_error == "raise":
                raise
            return []

    def delete_annotations_on_frame(self, current_frame: int):
        for annotation_list in (self.raw_thorax_positions, self.waggle_starts):
            idx = Annotations.get_annotation_index_for_frame(
                annotation_list, current_frame
            )
            if idx is not None:
                del annotation_list[idx]


# Utility functions that might be needed

def get_output_filename(filepath: str) -> str:
    # Extract the base file name without the extension
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    # Get the directory of the input file
    directory = os.path.dirname(filepath)
    # Create the output filename in the same directory
    output_filename = os.path.join(directory, f"{base_name}_waggle_annotations.csv")
    return output_filename


def get_csv_writer_options() -> dict:
    return dict(quoting=csv.QUOTE_ALL, quotechar='"', delimiter=",")