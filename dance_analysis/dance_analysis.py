#!/usr/bin/env python3
"""!
Overview
--------

This is a small Python tool to annotate videos with either arrows (left click) or just points (right click).

Each annotation session for a video is considered to be belong to "one dance" and during that session only the data for that dance can be modified.
Usually, the first frame of a waggle run is annotated with a vector by left-clicking on the bee's thorax and dragging the arrow in the direction of the individual's orientation.
The last frame of the waggle (the end of the waggle run) is marked by right-clicking the thorax of the bee.

As each left-click annotation should be paired with a right-click annotation,
matching the arrow to the next thorax annotation therefore also gives the duration of a waggle run.

Once the dance is fully annotated, the video can be closed (e.g. with Q) and the data will be saved automatically.
To fully erase already saved annotations, it will be necessary to delete the corresponding lines from the output .csv file.
To erase and reset the current annotation session, you can press R.

To anotate another dance in the video, simply reopen the file from the file selector. Already annotated dances will be shown in grey, making it easier to spot yet unmarked waggles.

In each frame of the video, only one annotation of either type (arrow or point) can exist. So to change an annotation, go to the corresponding frame and left-click or right-click again.
Annotations in a specific frame can be removed with the Backspace (<-, above Enter) key.

Keymapping
-----------

| Key                       | Description                                                                                 | 
|---------------------------|---------------------------------------------------------------------------------------------|
| Left click                | Create new arrow or update existing one (hold pressed down to specify direction).           |
| Right click               | Create new point or update existing one. |
| Space                     | Pause/unpause playback. |
| a                         | Go one frame back in time. |
| d                         | Go one frame forward in time. |
| w or shift+d              | Go one second forward in time. |
| s or shift+a              | Go one second backward in time. |
| 1 or ctrl+a               | Go five seconds back in time. |
| 3 or ctrl+d               | Go five seconds forward in time. |
| f                         | Jump to the last annotation in the video. |
| +                         | Increase replay speed. |
| -                         | Decrease replay speed. |
| h                         | Hide/show all annotations except for the most recent ones. |
| c                         | Switch through different contrast improvement methods. |
| r                         | Delete all current annotations and go to start of video. |
| q                         | Save current annotations and close video. |
| x or backspace            | Delete annotations in current frame.  |

"""

import re
import sys
import csv
import os.path
import argparse
import dataclasses
import time

import cv2 as cv
import numpy as np
import pandas as pd
import skimage


try:
    import av
except ImportError as e:
    av = None

# Global FPS setting.
# -1 means the software will query the video metadata.
VIDEO_FPS = -1


def get_output_filename(filepath):
    # Extract the base file name without the extension
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    # Get the directory of the input file
    directory = os.path.dirname(filepath)
    # Create the output filename in the same directory
    output_filename = os.path.join(directory, f"{base_name}_waggle_annotations.csv")
    return output_filename


def get_csv_writer_options():
    return dict(quoting=csv.QUOTE_ALL, quotechar='"', delimiter=",")


@dataclasses.dataclass
class AnnotatedPosition:
    frame: int
    x: int
    y: int

    # Angle is annotated in vector notation in the same coordinate
    # system as the x,y coordinates. This hopefully will prevent some
    # confusion about the origin of an angle in image coordinates.
    u: float = 0.0
    v: float = 0.0


class Annotations:
    @staticmethod
    def get_annotation_index_for_frame(list, frame):
        index = [idx for idx, position in enumerate(
            list) if position.frame == frame]
        assert len(index) < 2
        if len(index) == 0:
            return None
        return index[0]

    def __init__(self):
        self.raw_thorax_positions = []
        self.waggle_starts = []

    def update_thorax_position(self, frame, x, y):
        existing_index = Annotations.get_annotation_index_for_frame(
            self.raw_thorax_positions, frame
        )
        if existing_index is not None:
            self.raw_thorax_positions[existing_index].x = x
            self.raw_thorax_positions[existing_index].y = y
        else:
            self.raw_thorax_positions.append(AnnotatedPosition(frame, x, y))

    def update_waggle_start(self, frame, x, y, u=np.nan, v=np.nan):
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

    def update_waggle_direction(self, frame, to_x, to_y):
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
        self.__init__()

    def calculate_min_max_thorax_distance_to_actuator(self, actuator):
        if len(self.raw_thorax_positions) == 0 or not actuator:
            return None
        actuator = np.array(actuator)
        thorax_xy = np.array([(p.x, p.y) for p in self.raw_thorax_positions])
        distances = np.linalg.norm(thorax_xy - actuator, axis=1)
        return np.around(distances.min(), decimals=2), np.around(
            distances.max(), decimals=2
        )

    def get_maximum_annotated_frame_index(self):
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
        except Exception:
            frame = None
        return frame

    def is_empty(self):
        return self.get_maximum_annotated_frame_index() is None

    @staticmethod
    def load(filepath, on_error="print"):
        def parse_string_list(values):
            if isinstance(values, list):
                values = map(ast.literal_eval, values)
                values = list(itertools.chain(*values))
            else:
                values = ast.literal_eval(values)
            return values

        # Try to read in old annotated data for this video, but don't
        # fail fatally if anything happens.
        try:
            all_annotations = []

            import ast
            import pathlib
            import itertools

            # from collections import defaultdict

            annotations_df = pd.read_csv(
                get_output_filename(filepath),
                decimal=".",
                **get_csv_writer_options(),
            )

            def to_leaf_name(path):
                return pathlib.Path(path).name

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

    def delete_annotations_on_frame(self, current_frame):
        for annotation_list in (self.raw_thorax_positions, self.waggle_starts):
            idx = Annotations.get_annotation_index_for_frame(
                annotation_list, current_frame
            )
            if idx is not None:
                del annotation_list[idx]


class FileSelectorUI:
    def __init__(self, root_path, on_filepath_selected):
        self.root_path = root_path
        self.on_filepath_selected = on_filepath_selected
        self.index_map = {}
        self.instructions_frame = None 

    def collect_files(self):
        import os
        import pathlib

        # List of common video file extensions
        video_extensions = (
            '.mp4', '.avi', '.h264', '.mov', '.mkv',
            '.mpeg', '.mpg', '.wmv', '.flv', '.m4v',
            '.3gp', '.3g2'
        )

        for root, dirs, files in os.walk(self.root_path):
            for name in files:
                # Check if the file ends with any of the video extensions (case-insensitive)
                if name.lower().endswith(video_extensions):
                    filepath = os.path.join(root, name)
                    yield filepath, name

    def load_old_annotation_infos(self, filepath):
        annotations = Annotations.load(filepath, on_error="silent")

        n_waggle_starts, n_thorax_points, max_annotated_frame = 0, 0, 0
        n_dances = 0

        if annotations:
            n_waggle_starts = sum(len(a.waggle_starts) for a in annotations)
            n_thorax_points = sum(len(a.raw_thorax_positions)
                                  for a in annotations)
            max_annotated_frame = max(
                int(a.get_maximum_annotated_frame_index() or 0) for a in annotations
            )
            n_dances = len(annotations)

        return dict(
            n_waggle_starts=n_waggle_starts,
            n_thorax_points=n_thorax_points,
            max_annotated_frame=max_annotated_frame,
            n_dances=n_dances,
        )

    def edit_video_file(self, event):
        row = self.pt.get_row_clicked(event)
        raw_table = self.pt.model.df
        idx = raw_table.index[row]
        filepath = self.index_map[idx]

        self.on_filepath_selected(
            filepath, **self.get_additional_processing_kwargs())

        # Update row.
        annotation_infos = self.load_old_annotation_infos(filepath)
        for key, value in annotation_infos.items():
            raw_table.at[idx, key] = value

        self.pt.redraw()

    def create_instructions_table(self, parent):
        # Create a frame for the instructions table
        import tkinter as tk
        self.instructions_frame = tk.Frame(parent)
        
        # Title for the instructions
        title_label = tk.Label(self.instructions_frame, text="Instructions and Key Mappings", font=('Arial', 14, 'bold'))
        title_label.pack(pady=(5, 10))

        # Table Frame
        table_frame = tk.Frame(self.instructions_frame)
        table_frame.pack(fill="x")

        # Headers
        headers = ["Key", "Description"]
        for col_num, header in enumerate(headers):
            label = tk.Label(table_frame, text=header, font=('Arial', 12, 'bold'), borderwidth=1, relief="solid", padx=5, pady=5)
            label.grid(row=0, column=col_num, sticky="nsew")

        # Instructions data
        instructions = [
            ("Left click", "Create new arrow or update existing one (hold pressed down to specify direction)."),
            ("Right click", "Create new point or update existing one."),
            ("Space", "Pause/unpause playback."),
            ("a", "Go one frame back in time."),
            ("d", "Go one frame forward in time."),
            ("w or shift+d", "Go one second forward in time."),
            ("s or shift+a", "Go one second backward in time."),
            ("1 or ctrl+a", "Go five seconds back in time."),
            ("3 or ctrl+d", "Go five seconds forward in time."),
            ("f", "Jump to the last annotation in the video."),
            ("+", "Increase replay speed."),
            ("-", "Decrease replay speed."),
            ("h", "Hide/show all annotations except for the most recent ones."),
            ("c", "Switch through different contrast improvement methods."),
            ("r", "Delete all current annotations and go to start of video."),
            ("q", "Save current annotations and close video."),
            ("x or backspace", "Delete annotations in current frame."),
        ]

        # Fill the table with instructions
        for row_num, (key, desc) in enumerate(instructions, start=1):
            key_label = tk.Label(table_frame, text=key, font=('Arial', 12), borderwidth=1, relief="solid", padx=5, pady=5)
            key_label.grid(row=row_num, column=0, sticky="nsew")
            desc_label = tk.Label(table_frame, text=desc, font=('Arial', 12), borderwidth=1, relief="solid", padx=5, pady=5)
            desc_label.grid(row=row_num, column=1, sticky="nsew")

        # Adjust column widths
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_columnconfigure(1, weight=5)

    def toggle_instructions(self):
        if self.instructions_frame.winfo_ismapped():
            self.instructions_frame.pack_forget()
        else:
            self.instructions_frame.pack(fill="x", pady=10)

    def show(self):
        import tkinter as tk
        import pandastable

        self.index_map = dict()

        table = []
        for idx, (filepath, filename) in enumerate(self.collect_files()):
            self.index_map[idx] = filepath

            metadata = dict(index=idx, filename=filename)
            metadata = {**metadata, **self.load_old_annotation_infos(filepath)}

            table.append(metadata)

        if not table:
            from tkinter import messagebox
            messagebox.showerror("No files found", "No video files available. Stopping.")
            exit(1)

        table = pd.DataFrame(table).sort_values("filename")
        table.set_index("index", inplace=True)

        self.root = tk.Tk()
        self.root.title("Available videos")
        # Toggle Button for instructions
        self.create_instructions_table(self.root)  # first create the table object
        toggle_button = tk.Button(self.root, text="Show/Hide Instructions", command=self.toggle_instructions)
        toggle_button.pack(pady=10)

        self.checkbox_frame = tk.Frame(self.root, width=800, height=100)
        self.checkbox_frame.pack(fill="x", expand=True)

        self.checkboxes = []
        for idx, (argname, description, default_value) in enumerate(
            [
                ("start_paused", "Start Paused", 0),
                ("use_pyav", "Use PyAV library", 0)
            ]
        ):
            cb_var = tk.IntVar(value=default_value)
            cb = tk.Checkbutton(self.checkbox_frame,
                                text=description, variable=cb_var)
            cb.pack(padx=5, pady=15, side=tk.LEFT)
            self.checkboxes.append((argname, cb_var))

        self.table_frame = tk.Frame(self.root, width=800, height=600)
        self.table_frame.pack(fill="x", expand=True)
        self.pt = pandastable.Table(
            self.table_frame, dataframe=table, cellwidth=100, width=800
        )
        self.pt.bind("<Double-Button-1>", self.edit_video_file)
        self.pt.show()

        # Show the instructions table initially
        self.instructions_frame.pack(fill="x", pady=10)


        self.table_frame.mainloop()



    def get_additional_processing_kwargs(self):
        kwargs = {arg: (variable.get() == 1)
                  for (arg, variable) in self.checkboxes}
        print(kwargs)
        return kwargs

class PyavVideoCapture:
    def __init__(self, filename, frame_count, video_fps):
        self.capture = av.open(filename)
        self.length = frame_count
        self.fps = video_fps
        self.frame_index = 0

        # Select the video stream from the container
        self.video_stream = next((s for s in self.capture.streams if s.type == 'video'), None)
        
        if self.video_stream is None:
            raise ValueError("No video stream found in the file.")        
        else:
            print('Found video stream',  self.video_stream)

    def read(self):
        if self.frame_index < 0 or self.frame_index > self.length:
            print('Returning false due to invalid frame index')
            return False, None
        try:
            # Decode the next frame from the video stream
            frame = next(self.capture.decode(0))  # added index of 0, because just video is being opend
            self.frame_index += 1
        except StopIteration:
            # End of the video stream
            print('End of video stream')
            return False, None
        except av.AVError as e:
            # Handle specific PyAV errors
            print(f"Decoding error: {e}")
            return False, None
        except Exception as e:
            # Handle general errors
            print(f"Unexpected error: {e}")
            return False, None
        
        return True, frame.to_ndarray(format="bgr24")
    
    def get(self, property):
        if property == cv.CAP_PROP_POS_FRAMES:
            return self.frame_index
        if property == cv.CAP_PROP_FRAME_COUNT:
            return self.length
        raise ValueError("Unsupported property")
    
    def set(self, property, value):
        if property == cv.CAP_PROP_POS_FRAMES:
            old_frame_index = self.frame_index
            self.frame_index = max(0, value)

            if self.frame_index != old_frame_index:
                print(f"Frame index from {old_frame_index} to {self.frame_index}.")
                self.seek(self.frame_index)
            return
        raise ValueError("Unsupported property")
    
    def seek(self, frame_index):
        print("Seeking to {}".format(frame_index))
        time_base = self.capture.streams.video[0].time_base
        framerate = self.capture.streams.video[0].average_rate
        
        sec = int(frame_index / framerate) # round down to nearest key frame
        self.capture.seek(sec * 1000000, whence='time', backward=True)

        keyframe = next(self.capture.decode(video=0))
        keyframe_frame_index = int(keyframe.pts * time_base * framerate)

        for _ in range(keyframe_frame_index, frame_index - 1):
            _ = next(self.capture.decode(video=0))

        self.frame_index = frame_index

    def release(self):
        del self.capture
        
class VideoCaptureCache:
    def __init__(self, capture, cache_size=None, verbose=False):
        global VIDEO_FPS

        if cache_size is None:
            cache_size = VIDEO_FPS * 3

        self.verbose = verbose
        self.capture = capture
        self.cache_size = cache_size
        self.cache = dict()
        self.age_counter = 0
        self.current_frame = int(self.capture.get(cv.CAP_PROP_POS_FRAMES))
        self.last_read_frame = -1

    def delete_oldest(self):
        if len(self.cache) == 0:
            return

        # Find the oldest cached frame
        oldest_key = min(self.cache, key=lambda k: self.cache[k][0])
        del self.cache[oldest_key]

    def ensure_max_cache_size(self):
        while len(self.cache) > self.cache_size:
            if self.verbose:
                print(f"Dropping frame from cache ({len(self.cache)} / {self.cache_size})")
            self.delete_oldest()

    def get_current_frame(self):
        return self.current_frame

    def set_current_frame(self, to_frame):
        self.current_frame = to_frame

    def read(self):
        valid, frame = None, None

        if self.current_frame in self.cache:
            _, valid, frame = self.cache[self.current_frame]
            if self.verbose:
                print("Cache hit ({})".format(self.current_frame))
        else:
            if self.last_read_frame != self.current_frame - 1:
                if self.verbose:
                    print("Manual seek ({})".format(self.current_frame))
                self.capture.set(cv.CAP_PROP_POS_FRAMES, self.current_frame)
            self.last_read_frame = self.current_frame
            try:
                valid, frame = self.capture.read()
            except Exception as e:
                valid = False
                frame = None
                print(f"Could not read next frame: {repr(e)}")

            if self.verbose:
                print("Cache miss ({})".format(self.current_frame))

        self.cache[self.current_frame] = self.age_counter, valid, frame
        self.current_frame += 1
        self.age_counter += 1
        self.ensure_max_cache_size()

        return valid, frame
    
class FramePostprocessingPipeline:

    CROP = 0
    CONTRAST = 1
    MAX_STEPS = 2

    class PipelineStep:
        def process(self, frame):
            return frame
        def transform_coordinates_video_to_screen(self, coords):
            return coords
        def transform_coordinates_screen_to_video(self, coords):
            return coords

    class CropRectAroundCursor(PipelineStep):

        def __init__(self, frame, mouse_position):
            
            self.mouse_position = mouse_position
            self.frame_shape = frame.shape
            self.zoom_factor = 2

            self.update_zoom(self.mouse_position, self.frame_shape, self.zoom_factor)

        def update_zoom(self, mouse_position, frame_shape, zoom):
            mouse_x, mouse_y = mouse_position

            scale = 1.0 / zoom
            self.x, self.y = 0, 0
            self.h, self.w  = int(scale * frame_shape[0]), int(scale * frame_shape[1])

            self.x = mouse_x - self.w // 2
            self.y = mouse_y - self.h // 2

            right_side = self.x + self.w
            bottom_side = self.y + self.h

            if right_side > frame_shape[1]:
                self.x -= (right_side - frame_shape[1])
            
            if bottom_side > frame_shape[0]:
                self.y -= (bottom_side - frame_shape[0])

            if self.x < 0:
                self.x = 0
            if self.y < 0:
                self.y = 0


        def process(self, frame):

            frame = frame[(self.y):(self.y + self.h),
                          (self.x):(self.x + self.w)]
            
            return frame
        
        def transform_coordinates_video_to_screen(self, coords):
            x, y = coords
            return (x - self.x, y - self.y)
        
        def transform_coordinates_screen_to_video(self, coords):
            x, y = coords
            return (x + self.x, y + self.y)
        
        def adjust_zoom(self, direction):
            old_zoom_factor = self.zoom_factor

            if direction > 0:
                self.zoom_factor *= 1.1
            else:
                self.zoom_factor /= 1.1
            
            self.zoom_factor = max(self.zoom_factor, 1.0)
            self.zoom_factor = min(self.zoom_factor, 8.0)

            self.update_zoom(self.mouse_position, self.frame_shape, self.zoom_factor)

            return self.zoom_factor != old_zoom_factor

        
    class ContrastNormalizationFast(PipelineStep):
        
        def __init__(self, frame):

            H, W = frame.shape[:2]
            mid_y, mid_x = H // 2, W // 2
            crop_h, crop_w = H // 3, W // 3
            center = frame[(mid_y - crop_h):(mid_y + crop_h), (mid_x - crop_w):(mid_x + crop_w)]

            data = center.flatten()
            self.min, self.max = np.percentile(data, (5, 95))
            self.min = float(self.min)
            self.max = float(self.max) - self.min

        def process(self, frame):

            # Use in-place operations for a slightly better performance.
            frame = frame.astype(np.float32)
            np.subtract(frame, self.min, out=frame)
            np.divide(frame, self.max / 255.0, out=frame)
            np.clip(frame, 0, 255, out=frame)
            frame = frame.astype(np.uint8)

            return frame
        
    class ContrastNormalizationFastCenter(PipelineStep):
        
        def __init__(self, frame):
            H, W = frame.shape[:2]
            mid_y, mid_x = H // 2, W // 2
            crop_h, crop_w = H // 6, W // 6
            center = frame[(mid_y - crop_h):(mid_y + crop_h), (mid_x - crop_w):(mid_x + crop_w)]

            self.min = center.min()
            self.max = center.max() - self.min
            if self.max <= 0:
                self.max = 1

        def process(self, frame):

            # Use in-place operations for a slightly better performance.
            frame = frame.astype(np.float32)
            np.subtract(frame, self.min, out=frame)
            np.divide(frame, self.max / 255.0, out=frame)
            np.clip(frame, 0, 255, out=frame)
            frame = frame.astype(np.uint8)

            return frame
        
    class ContrastHistogramEqualization(PipelineStep):

        def __init__(self, **kwargs):
            pass

        def process(self, frame):
            import skimage.exposure

            frame = skimage.exposure.equalize_hist(frame)

            return frame
    
    def __init__(self):
        self.steps = [None] * FramePostprocessingPipeline.MAX_STEPS
        self.options_map = [0] * len(self.steps)

    def process(self, frame):
        for step in self.steps:
            if step is not None:
                frame = step.process(frame)
        return frame
    
    def select_next_option(self, option_type, options, force_index=None, **kwargs):
        options = [None] + options

        if force_index is None:
            self.options_map[option_type] = (self.options_map[option_type] + 1) % len(options)
        else:
            if self.options_map[option_type] == force_index:
                return
            
            self.options_map[option_type] = force_index

        new_step = options[self.options_map[option_type]]
        self.steps[option_type] = new_step(**kwargs) if new_step is not None else None

    def select_next_cropping(self, **kwargs):
        options = [FramePostprocessingPipeline.CropRectAroundCursor]
        self.select_next_option(FramePostprocessingPipeline.CROP, options, **kwargs)
    
    def select_first_cropping(self, **kwargs):
        self.select_next_cropping(force_index=1, **kwargs)

    def disable_cropping(self, **kwargs):
        self.select_next_cropping(force_index=0, **kwargs)

    def select_next_contrast_postprocessing(self, **kwargs):
        options = [FramePostprocessingPipeline.ContrastNormalizationFastCenter,
                   FramePostprocessingPipeline.ContrastNormalizationFast,
                   FramePostprocessingPipeline.ContrastHistogramEqualization]
        self.select_next_option(FramePostprocessingPipeline.CONTRAST, options, **kwargs)

    def transform_coordinates_video_to_screen(self, coords):
        for step in self.steps:
            if step is not None:
                coords = step.transform_coordinates_video_to_screen(coords)
        return coords
    
    def transform_coordinates_screen_to_video(self, coords):
        for step in self.steps:
            if step is not None:
                coords = step.transform_coordinates_screen_to_video(coords)
        return coords
    
    def adjust_zoom(self, direction):
        was_handled = False
        step = self.steps[FramePostprocessingPipeline.CROP]
        if step is not None:
            handled_now = step.adjust_zoom(direction)
            was_handled = was_handled or handled_now

        return was_handled

def draw_template(img, current_time=None):

    video_height = img.shape[0]

    font = cv.FONT_HERSHEY_SIMPLEX

    if current_time is not None:
        for scale, color in \
            [
                (3.0, (50, 50, 50)),
                (1.0, (255, 255, 255))
            ]:
            img = cv.putText(
                img,
                "time {:2.2f} s ".format(current_time),
                (0, video_height - 24),
                font,
                1,
                color,
                int(2 * scale),
                cv.LINE_AA,
            )

    return img


def draw_bee_positions(
    img,
    annotations,
    current_frame,
    is_old_annotations=False,
    hide_past_annotations=False,
    frame_postprocessing_pipeline=None
):
    colormap = dict(
        thorax_position=(0, 255, 0),
        thorax_position_100_frames=(0, 0, 255),
        waggle_start=(0, 255, 255),
    )
    if is_old_annotations:
        colormap = dict(
            thorax_position=(200, 200, 200),
            thorax_position_100_frames=(200, 200, 255),
            waggle_start=(200, 255, 255),
        )

    if annotations.raw_thorax_positions:
        last_marker_frame = max(
            [p.frame for p in annotations.raw_thorax_positions])

    for position in annotations.raw_thorax_positions:
        is_last_marker = position.frame == last_marker_frame
        is_in_current_frame = current_frame == position.frame

        if (not is_last_marker and not is_in_current_frame) and hide_past_annotations:
            continue
        
        (x, y) = (position.x, position.y)
        if frame_postprocessing_pipeline is not None:
            (x, y) = frame_postprocessing_pipeline.transform_coordinates_video_to_screen((x, y))

        radius = 5 if not is_in_current_frame else 10
        img = cv.circle(
            img, (x, y), radius, colormap["thorax_position"], 2
        )

        # We want to mark the bee 100 frames after the last thorax marking,
        # so highlight the last one at the 100 frames mark.
        if is_last_marker and current_frame > position.frame:
            size = radius
            if position.frame == current_frame - 100:
                size = radius * 6
            img = cv.drawMarker(
                img,
                (x, y),
                colormap["thorax_position_100_frames"],
                markerType=cv.MARKER_STAR,
                markerSize=size,
            )

    if annotations.waggle_starts:
        last_waggle_start_frame = max(
            [p.frame for p in annotations.waggle_starts])

    for position in annotations.waggle_starts:
        is_last_marker = position.frame == last_waggle_start_frame
        is_in_current_frame = current_frame == position.frame

        if (not is_last_marker and not is_in_current_frame) and hide_past_annotations:
            continue
        
        (x, y) = (position.x, position.y)
        if frame_postprocessing_pipeline is not None:
            (x, y) = frame_postprocessing_pipeline.transform_coordinates_video_to_screen((x, y))

        radius = 2 if not is_in_current_frame else 5
        length = 25 if not is_in_current_frame else 50
        img = cv.circle(
            img, (x, y), radius, colormap["waggle_start"], 2
        )

        if not pd.isnull(position.u) and not (
            np.abs(position.u) < 1e-4 and np.abs(position.v) < 1e-4
        ):
            direction = np.array([position.u, position.v])
            direction = (direction / np.linalg.norm(direction)) * length
            direction = direction.astype(int)

            img = cv.arrowedLine(
                img,
                (int(x - direction[0]), int(y - direction[1])),
                (int(x + direction[0]), int(y + direction[1])),
                colormap["waggle_start"],
                thickness=1,
            )

    return img


def setup(cap, frame_change_callback=lambda x: x):
    cv.namedWindow("Frame", cv.WINDOW_GUI_NORMAL)
    cv.createTrackbar("Speed", "Frame", VIDEO_FPS, 3 * VIDEO_FPS, lambda x: x)

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cv.createTrackbar("Frame", "Frame", 0, total_frames, frame_change_callback)


def calculate_time(current_frame, video_fps):
    frame_timestamp_sec = current_frame / video_fps
    return round(frame_timestamp_sec, 2)


def output_data(annotations: Annotations, filepath):

    # Write to csv
    header = [
        "video_name",
        "thorax_positions",
        "thorax_frames",
        "waggle_start_positions",
        "waggle_start_frames",
        "waggle_directions",
    ]
    thorax_xy = [(p.x, p.y) for p in annotations.raw_thorax_positions]
    thorax_frames = [p.frame for p in annotations.raw_thorax_positions]
    waggle_xy = [(p.x, p.y) for p in annotations.waggle_starts]
    waggle_frames = [p.frame for p in annotations.waggle_starts]
    waggle_directions = [(p.u, p.v) for p in annotations.waggle_starts]
    data = [
        filepath,
        thorax_xy,
        thorax_frames,
        waggle_xy,
        waggle_frames,
        waggle_directions,
    ]

    output_filepath = get_output_filename(filepath)
    is_first_entry = not os.path.exists(output_filepath)

    with open(output_filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, **get_csv_writer_options())

        if is_first_entry:
            writer.writerow(header)
        writer.writerow(data)


def calc_frames_to_move(k32: int, debug: bool = False) -> int:
    """ from extended keycode, select nframes to advance or rewind."""
    # hmm modifiers are actually 17, 19, 20 bits away
    ctrl = (k32 & (1 << 18)) >> 18  # noqa: E221
    shift = (k32 & (1 << 16)) >> 16
    alt = (k32 & (1 << 19)) >> 19  # noqa: E221
    modifiers = f"{ctrl:0x} {shift:0x} {alt:0x}"
    if k32 != -1 and debug:
        key = k32 & 0xff
        print(
            "\033[34m"
            f"KEY: {k32} 0x{k32:2x} |{key} 0x{key:2x} |{modifiers}|"
            "\033[0m"
        )
    nframes = 1
    if shift:
        nframes = int(VIDEO_FPS * 1)
    elif ctrl:
        nframes = int(VIDEO_FPS * 5)
    elif alt:
        nframes = int(5)

    return nframes


def fit_image_to_aspect_ratio(image, target_width, target_height):
    """
    Fits the aspect ratio of an image to fit in given width/height constraints.
    The aspect ratio is changed by adding a black border at the bottom or right of the image;
    this retains the pixel coordinates so that annotation coordinates do not have to be adjusted.
    """

    image_height, image_width, _ = image.shape

    target_aspect = target_width / target_height
    image_aspect = image_width / image_height

    bottom_border = 0
    right_border = 0

    if target_aspect == image_aspect:
        pass
    elif target_aspect > image_aspect:
        right_border = int(image_height * target_aspect) - image_width
    else:
        bottom_border = int(image_width / target_aspect) - image_height

    if bottom_border > 0 or right_border > 0:
        image = cv.copyMakeBorder(
            image, 0, bottom_border, 0, right_border, cv.BORDER_CONSTANT
        )

    return image


def do_video(
    filepath: str,
    debug: bool = False,
    start_paused: bool = False,
    enable_fit_image_to_window: bool = True,
    use_pyav: bool = False
):
    global VIDEO_FPS
    annotations = Annotations()
    old_annotations = Annotations.load(filepath)

    cap = cv.VideoCapture(filepath)

    total_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    video_reader_fps = cap.get(cv.CAP_PROP_FPS)

    if use_pyav:
        if av is not None:
            print("Using PyAV library for decoding.")
            del cap
            cap = PyavVideoCapture(filepath, total_frames, video_reader_fps)
        else:
            print("PyAV library not found, using OpenCV.")

    if VIDEO_FPS == -1:
        if video_reader_fps > 0:
            VIDEO_FPS = int(video_reader_fps + 0.5)
        else:
            VIDEO_FPS = 60

    capture_cache = VideoCaptureCache(cap)
    original_frame_image = None
    current_frame = capture_cache.get_current_frame()

    assert current_frame == 0
    is_in_pause_mode = start_paused
    is_in_draw_vector_mode = False
    hide_past_annotations = False
    frame_postprocessing_pipeline = FramePostprocessingPipeline()

    def move_frame_count(offset, target_frame=None):
        nonlocal current_frame
        nonlocal cap
        nonlocal total_frames
        nonlocal original_frame_image

        if target_frame is None:
            target_frame = current_frame + offset

        current_frame = min(max(target_frame, 0), total_frames - 1)
        if is_in_pause_mode or (current_frame - capture_cache.get_current_frame()>1):
            capture_cache.set_current_frame(current_frame)

        # Clear current raw frame so we fetch a new one even if in pause mode.
        original_frame_image = None

    setup(cap, frame_change_callback=lambda f: move_frame_count(offset = 0, target_frame=f))

    # Set up mouse interaction callback.
    last_mouse_position = (0, 0)
    def on_mouse_event(event, x, y, flags, user_data):
        nonlocal current_frame, is_in_draw_vector_mode, is_in_pause_mode
        nonlocal last_mouse_position, frame_postprocessing_pipeline

        (x, y) = frame_postprocessing_pipeline.transform_coordinates_screen_to_video((x, y))

        if event == cv.EVENT_FLAG_RBUTTON:  # right click
            annotations.update_thorax_position(current_frame, x, y)
        elif event == cv.EVENT_LBUTTONDOWN:  # double click
            annotations.update_waggle_start(current_frame, x, y)
            is_in_draw_vector_mode = True
            # Force pause for drawing. Otherwise, the updates might
            # affect the wrong arrow.
            if not is_in_pause_mode:
                # Additional marker to True/False for "forced pause".
                is_in_pause_mode = 2
        elif event == cv.EVENT_MOUSEMOVE:
            last_mouse_position = (x, y)
            if is_in_draw_vector_mode:
                annotations.update_waggle_direction(current_frame, x, y)
        elif event == cv.EVENT_LBUTTONUP:
            is_in_draw_vector_mode = False
            if is_in_pause_mode == 2:
                is_in_pause_mode = False
            annotations.update_waggle_direction(current_frame, x, y)
        elif event == cv.EVENT_MOUSEWHEEL:
            is_zooming_in = flags > 0
            could_zoom_further = frame_postprocessing_pipeline.adjust_zoom(int(is_zooming_in))

            if is_zooming_in:
                if last_raw_frame is not None:
                    frame_postprocessing_pipeline.select_first_cropping(frame=last_raw_frame, mouse_position=last_mouse_position)
            else:
                if not could_zoom_further:
                    frame_postprocessing_pipeline.disable_cropping()

    # We had to create the window with a flag that disables
    # right-click context menu.
    cv.setMouseCallback("Frame", on_mouse_event)

    processing_time_ms_avg = 10.0
    processing_time_last_start = time.perf_counter()
    last_raw_frame = None

    while True:
        try:
            speed_fps = cv.getTrackbarPos("Speed", "Frame")
        except Exception:
            # Might fail when window is already deconstructed.
            # But then, the application is being terminated anyway.
            print("Stopping.")
            break
        has_valid_frame = True

        # Fetch a new frame if either the speed is > 0 or we are not
        # currently pausing the video playback.
        if (speed_fps > 0 and not is_in_pause_mode) or original_frame_image is None:
            # Make sure to read the current frame number only BEFORE fetching an image.
            # cap.read() will advance the frame number after reading.
            current_frame = capture_cache.get_current_frame()
            has_valid_frame, original_frame_image = capture_cache.read()
            if original_frame_image is not None:
                last_raw_frame = original_frame_image

        if not has_valid_frame:
            move_frame_count(-1)
            is_in_pause_mode = True
            continue

        frame = original_frame_image.copy()
        frame = frame_postprocessing_pipeline.process(frame)

        frame = draw_template(
            frame,
            current_time=calculate_time(current_frame, video_reader_fps),
        )
        frame = draw_bee_positions(
            frame,
            annotations,
            current_frame=current_frame,
            hide_past_annotations=hide_past_annotations,
            frame_postprocessing_pipeline=frame_postprocessing_pipeline
        )
        if old_annotations and not hide_past_annotations:
            for old_annotations_session in old_annotations:
                frame = draw_bee_positions(
                    frame,
                    old_annotations_session,
                    current_frame=current_frame,
                    is_old_annotations=True,
                    frame_postprocessing_pipeline=frame_postprocessing_pipeline
                )

        cv.setTrackbarPos("Frame", "Frame", int(current_frame))

        last_processing_timestamp = processing_time_last_start
        current_processing_timestamp = time.perf_counter() # reset counter
        processing_duration_ms = 1000.0 * (current_processing_timestamp - last_processing_timestamp)
        processing_time_ms_avg = (0.9 * processing_time_ms_avg) + (0.1 * processing_duration_ms)

        if speed_fps <= 0 or is_in_pause_mode:
            # Arbitrary delay > 0, because we need to update the UI
            # even in pause mode.
            delay_ms = 50
        else:
            delay_ms = max(1, int(1000 / speed_fps - processing_time_ms_avg))

        k32 = cv.waitKeyEx(delay_ms)
        key = k32 & 0xff
        nframes = calc_frames_to_move(k32, debug)

        processing_time_last_start = time.perf_counter() # reset counter after sleep

        if key == ord("q"):
            # Press q to exit video
            break
        elif key == ord(" "):
            # Spacebar as pause button
            is_in_pause_mode = not is_in_pause_mode
        elif key == ord("r"):
            # Press r to restart video (and delete bee/stop positions)
            capture_cache.set_current_frame(0)
            original_frame_image = None
            annotations.clear()
        elif key in [ord("a"), ord("A"), 0x51]:
            # 0x51 is left arrow
            move_frame_count(-nframes)
        elif key in [ord("d"), ord("D"), 0x53]:
            # 0x53 is right arrow (weird, should be 37-40 LURD)
            move_frame_count(+nframes)
        elif key == ord("w"):
            move_frame_count(+VIDEO_FPS)
        elif key == ord("s"):
            move_frame_count(-VIDEO_FPS)
        elif key == ord("1"):
            move_frame_count(-5 * VIDEO_FPS)
        elif key == ord("3"):
            move_frame_count(+5 * VIDEO_FPS)
        elif key == ord("+"):
            cv.setTrackbarPos(
                "Speed", "Frame", min(
                    (speed_fps // 10) * 10 + 10, 3 * VIDEO_FPS)
            )
        elif key == ord("-"):
            cv.setTrackbarPos("Speed", "Frame", max(
                (speed_fps // 10) * 10 - 10, 0))
        elif key == ord("h"):
            hide_past_annotations = not hide_past_annotations
        elif key == ord("c"):
            if last_raw_frame is not None:
                frame_postprocessing_pipeline.select_next_contrast_postprocessing(frame=last_raw_frame)
        elif key == ord("v"):
            if last_raw_frame is not None:
                frame_postprocessing_pipeline.select_next_cropping(frame=last_raw_frame, mouse_position=last_mouse_position)
            else:
                print("No image woot")
        elif key in (ord("x"), 8):  # 8 is backspace.
            annotations.delete_annotations_on_frame(current_frame)
        elif key == ord("f"):
            max_frame_new = annotations.get_maximum_annotated_frame_index()
            if old_annotations:
                max_frame_old = max(
                    int(a.get_maximum_annotated_frame_index() or 0) for a in old_annotations
                )              
            else:
                max_frame_old = None
            max_frames = [f for f in (max_frame_old,max_frame_new) if f is not None]
            max_frame = max(max_frames) if len(max_frames)>0 else None
            if max_frame is not None:
                move_frame_count(offset=0, target_frame=max_frame)

        if enable_fit_image_to_window:
            try:
                _, _, window_width, window_height = cv.getWindowImageRect(
                    "Frame")
                if window_width > 0 and window_height > 0:
                    frame = fit_image_to_aspect_ratio(
                        frame, window_width, window_height
                    )
            except Exception as e:
                print("Could not fit image to window: {}".format(str(e)))

        cv.imshow("Frame", frame)
    
    import gc
    gc.collect()
    cap.release()
    cv.destroyAllWindows()

    # store dataset in file
    if not annotations.is_empty():
        output_data(annotations, filepath)
    else:
        print("No annotations to save.")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "-p", "--path", type=str, help="select path to video files", default='./'
    )
    args = parser.parse_args()


    folder_path = args.path

    ui = FileSelectorUI(
        folder_path,
        on_filepath_selected=lambda path, **kwargs: do_video(
            path, args.debug, **kwargs
        ),
    )
    ui.show()

if __name__ == "__main__":
    main()