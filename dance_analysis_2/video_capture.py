# video_capture.py

import cv2 as cv
import numpy as np
from typing import Optional, Tuple
import os
import csv
import platform
import tkinter as tk
from tkinter import HORIZONTAL, messagebox
from PIL import Image, ImageTk
import sys
import time
from .annotations import Annotations, pair_runs
from .pipeline import FramePostprocessingPipeline
from .utils import (
    calculate_time,
    fit_image_to_aspect_ratio,
    get_output_filename,
    get_csv_writer_options,
    open_video_capture,
)
from .theme import (
    BG, BG_PANEL, FG, FG_MUTED, ACCENT, ACCENT_BLUE, PLAYHEAD,
    ACCENT_BGR, ACCENT_BLUE_BGR, ACCENT_RED_BGR, dance_color_bgr, dance_color_hex,
)
from .timeline_widgets import (
    draw_diamond_marker, draw_playhead, draw_frame_ruler, RULER_HEIGHT, LABEL_HEIGHT,
    MARKER_SIZE, BlenderSlider,
)

try:
    import av
except ImportError:
    av = None


class PyavVideoCapture:
    def __init__(self, filename: str, frame_count: int, video_fps: float):
        self.capture = av.open(filename)
        self.length = frame_count
        self.fps = video_fps
        self.frame_index = 0

        # Select the video stream from the container
        self.video_stream = next((s for s in self.capture.streams if s.type == 'video'), None)

        if self.video_stream is None:
            raise ValueError("No video stream found in the file.")
        else:
            print('Found video stream:', self.video_stream)

        # Set up hardware acceleration based on the OS
        os_name = platform.system()
        
        if os_name == "Darwin":  # macOS
            self.video_stream.codec_context.options = {"hwaccel": "videotoolbox"}
            print("Hardware acceleration enabled with VideoToolbox (macOS)")
        elif os_name == "Linux":
            self.video_stream.codec_context.options = {"hwaccel": "cuda"}
            print("Hardware acceleration enabled with CUDA (Linux)")
        elif os_name == "Windows":
            self.video_stream.codec_context.options = {"hwaccel": "d3d11va"}
            print("Hardware acceleration enabled with DirectX Video Acceleration (Windows)")
        else:
            print("No compatible hardware acceleration found for this OS.")

        # Set thread count for parallel processing within FFmpeg
        self.video_stream.codec_context.thread_count = 4  # Adjust based on your CPU
        print(f"Thread count set to {self.video_stream.codec_context.thread_count}")

    def read(self):
        for packet in self.capture.demux(self.video_stream):
            for frame in packet.decode():
                if frame:
                    frame_ndarray = frame.to_ndarray(format="bgr24")
                    self.frame_index += 1
                    return frame_ndarray
        return None           

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.frame_index < 0 or self.frame_index > self.length:
            print('Returning false due to invalid frame index')
            return False, None
        try:
            # Decode the next frame from the video stream
            frame = next(self.capture.decode(video=0))
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
        # return True, frame.to_ndarray(format="bgr24")
        return True, frame.to_ndarray(format="bgr24").astype(np.uint8)

    def get(self, property: int) -> Optional[float]:
        # Handle the OpenCV-style property requests for PyAV
        if property == cv.CAP_PROP_POS_FRAMES:
            return self.frame_index
        if property == cv.CAP_PROP_FRAME_COUNT:
            return self.length
        if property == cv.CAP_PROP_FRAME_WIDTH:
            return self.video_stream.codec_context.width  # Width of the video
        if property == cv.CAP_PROP_FRAME_HEIGHT:
            return self.video_stream.codec_context.height  # Height of the video
        if property == cv.CAP_PROP_FPS:
            return self.video_stream.average_rate  # FPS of the video
        raise ValueError("Unsupported property")

    def set(self, property: int, value: int):
        if property == cv.CAP_PROP_POS_FRAMES:
            old_frame_index = self.frame_index
            self.frame_index = max(0, value)

            if self.frame_index != old_frame_index:
                print(f"Frame index from {old_frame_index} to {self.frame_index}.")
                self.seek(self.frame_index)
            return
        raise ValueError("Unsupported property")

    def seek(self, frame_index: int):
        print(f"Seeking to {frame_index}")
        time_base = self.capture.streams.video[0].time_base
        framerate = self.capture.streams.video[0].average_rate

        sec = int(frame_index / framerate)  # Round down to nearest key frame
        self.capture.seek(sec * 1_000_000, whence='time', backward=True)

        keyframe = next(self.capture.decode(video=0))
        keyframe_frame_index = int(keyframe.pts * time_base * framerate)

        for _ in range(keyframe_frame_index, frame_index - 1):
            _ = next(self.capture.decode(video=0))

        self.frame_index = frame_index

    def release(self):
        del self.capture


class VideoCaptureCache:
    def __init__(self, capture, cache_size: Optional[int] = None, verbose: bool = False, video_fps: float = 30.0):
        if cache_size is None:
            cache_size = int(video_fps * 3)

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

    def get_current_frame(self) -> int:
        return self.current_frame

    def set_current_frame(self, to_frame: int):
        self.current_frame = to_frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        valid, frame = None, None

        if self.current_frame in self.cache:
            _, valid, frame = self.cache[self.current_frame]
            if self.verbose:
                print(f"Cache hit ({self.current_frame})")
        else:
            if self.last_read_frame != self.current_frame - 1:
                if self.verbose:
                    print(f"Manual seek ({self.current_frame})")
                self.capture.set(cv.CAP_PROP_POS_FRAMES, self.current_frame)
            self.last_read_frame = self.current_frame
            try:
                valid, frame = self.capture.read()
            except Exception as e:
                valid = False
                frame = None
                print(f"Could not read next frame: {repr(e)}")

            if self.verbose:
                print(f"Cache miss ({self.current_frame})")

        self.cache[self.current_frame] = self.age_counter, valid, frame
        self.current_frame += 1
        self.age_counter += 1
        self.ensure_max_cache_size()

        return valid, frame

    def release(self):
        self.capture.release()


def draw_template(img: np.ndarray, current_time: Optional[float] = None) -> np.ndarray:
    video_height = img.shape[0]
    font = cv.FONT_HERSHEY_SIMPLEX

    if current_time is not None:
        for scale, color in [
            (3.0, (50, 50, 50)),
            (1.0, (255, 255, 255))
        ]:
            img = cv.putText(
                img,
                f"time {current_time:2.2f} s ",
                (0, video_height - 24),
                font,
                1,
                color,
                int(2 * scale),
                cv.LINE_AA,
            )
    return img


def draw_bee_positions(
    img: np.ndarray,
    annotations: Annotations,
    current_frame: int,
    hide_past_annotations: bool = False,
    frame_postprocessing_pipeline: Optional[FramePostprocessingPipeline] = None,
    show_all: bool = False,
    dance_number: Optional[int] = None,
    show_labels: bool = False,
) -> np.ndarray:
    # Markers from a previous session (old_annotations_list) use the same
    # vivid colors as live annotation - a muted/grey palette made them hard
    # to spot against already-grey/low-contrast footage.
    colormap = dict(
        thorax_position=ACCENT_BLUE_BGR,
        thorax_position_100_frames=ACCENT_RED_BGR,
        waggle_start=ACCENT_BGR,
    )

    paired_runs, unpaired_starts, unpaired_thorax = pair_runs(annotations)

    if annotations.raw_thorax_positions:
        last_marker_frame = max(p.frame for p in annotations.raw_thorax_positions)

    def draw_run_label(position, run_number):
        if not (show_labels and dance_number is not None):
            return
        x, y = position.x, position.y
        if frame_postprocessing_pipeline is not None:
            x, y = frame_postprocessing_pipeline.transform_coordinates_video_to_screen((x, y))
        text = f"D{dance_number}-R{run_number}"
        pos = (int(x) + 8, int(y) - 8)
        cv.putText(img, text, pos, cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv.LINE_AA)
        cv.putText(img, text, pos, cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)

    # Thorax/end markers: only visible while current_frame is within the
    # paired run's [start, end] range - unless show_all is set, which keeps
    # every run from the currently-active dance visible regardless of where
    # you've scrubbed to (past/saved dances stay range-scoped to cut clutter).
    for run_number, (start_position, end_position) in enumerate(paired_runs, start=1):
        if not show_all and not (start_position.frame <= current_frame <= end_position.frame):
            continue

        x, y = end_position.x, end_position.y
        if frame_postprocessing_pipeline is not None:
            x, y = frame_postprocessing_pipeline.transform_coordinates_video_to_screen((x, y))

        is_in_current_frame = current_frame == end_position.frame
        radius = 5 if not is_in_current_frame else 10
        img = cv.circle(
            img, (int(x), int(y)), radius, colormap["thorax_position"], 2
        )
        draw_run_label(end_position, run_number)

    # Any thorax positions left over without a matching start (shouldn't
    # happen going forward given the input ordering-constraint, but can occur
    # in older/atypical annotation files) keep the old always-ish-visible
    # behavior, so legacy data isn't silently hidden.
    for position in unpaired_thorax:
        is_last_marker = position.frame == last_marker_frame
        is_in_current_frame = current_frame == position.frame

        if (not is_last_marker and not is_in_current_frame) and hide_past_annotations:
            continue

        x, y = position.x, position.y
        if frame_postprocessing_pipeline is not None:
            x, y = frame_postprocessing_pipeline.transform_coordinates_video_to_screen((x, y))

        radius = 5 if not is_in_current_frame else 10
        img = cv.circle(
            img, (int(x), int(y)), radius, colormap["thorax_position"], 2
        )

        if is_last_marker and current_frame > position.frame:
            size = radius
            if position.frame == current_frame - 100:
                size = radius * 6
            img = cv.drawMarker(
                img,
                (int(x), int(y)),
                colormap["thorax_position_100_frames"],
                markerType=cv.MARKER_STAR,
                markerSize=size,
            )

    def draw_waggle_start(position):
        x, y = position.x, position.y
        if frame_postprocessing_pipeline is not None:
            x, y = frame_postprocessing_pipeline.transform_coordinates_video_to_screen((x, y))

        is_in_current_frame = current_frame == position.frame
        radius = 2 if not is_in_current_frame else 5
        length = 25 if not is_in_current_frame else 50
        img_local = cv.circle(
            img, (int(x), int(y)), radius, colormap["waggle_start"], 2
        )

        if not np.isnan(position.u) and not (
            np.abs(position.u) < 1e-4 and np.abs(position.v) < 1e-4
        ):
            direction = np.array([position.u, position.v])
            direction = (direction / np.linalg.norm(direction)) * length
            direction = direction.astype(int)

            img_local = cv.arrowedLine(
                img_local,
                (int(x - direction[0]), int(y - direction[1])),
                (int(x + direction[0]), int(y + direction[1])),
                colormap["waggle_start"],
                thickness=1,
            )
        return img_local

    # Waggle start markers: same [start, end] visibility rule when paired.
    for run_number, (start_position, end_position) in enumerate(paired_runs, start=1):
        if not show_all and not (start_position.frame <= current_frame <= end_position.frame):
            continue
        img = draw_waggle_start(start_position)
        draw_run_label(start_position, run_number)

    # A start still awaiting its end stays visible - nothing to bound it to yet.
    for position in unpaired_starts:
        img = draw_waggle_start(position)

    return img


def draw_reassign_overlay(
    img: np.ndarray,
    dances_numbered,
    frame_postprocessing_pipeline: Optional[FramePostprocessingPipeline] = None,
) -> np.ndarray:
    # Translucent colored ring at every marker, color-coded per dance, so
    # reassign mode shows at a glance which dance a run can be dropped onto.
    overlay = img.copy()
    for dance_number, dance in dances_numbered:
        color = dance_color_bgr(dance_number)
        for position in dance.waggle_starts + dance.raw_thorax_positions:
            x, y = position.x, position.y
            if frame_postprocessing_pipeline is not None:
                x, y = frame_postprocessing_pipeline.transform_coordinates_video_to_screen((x, y))
            cv.circle(overlay, (int(x), int(y)), 18, color, -1)
    return cv.addWeighted(overlay, 0.3, img, 0.7, 0)


def save_all_dances_for_video(dances, filepath: str) -> int:
    # Rewrites every row for this video (rather than appending) - needed
    # because reassign mode can move a run into or out of a dance that was
    # already saved in a previous session, which changes that row's
    # contents. Rows belonging to other videos in the same CSV are kept
    # exactly as they were, untouched and unparsed.
    header = [
        "video_name",
        "thorax_positions",
        "thorax_frames",
        "waggle_start_positions",
        "waggle_start_frames",
        "waggle_directions",
    ]
    rows = []
    for dance in dances:
        if dance.is_empty():
            continue
        rows.append([
            filepath,
            [(p.x, p.y) for p in dance.raw_thorax_positions],
            [p.frame for p in dance.raw_thorax_positions],
            [(p.x, p.y) for p in dance.waggle_starts],
            [p.frame for p in dance.waggle_starts],
            [(float(p.u), float(p.v)) for p in dance.waggle_starts],
        ])

    output_filepath = get_output_filename(filepath)
    target_leaf = os.path.basename(filepath)
    other_rows = []
    if os.path.exists(output_filepath):
        with open(output_filepath, newline="", encoding="utf-8") as f:
            existing_rows = list(csv.reader(f, **get_csv_writer_options()))
        other_rows = [
            row for row in existing_rows[1:]
            if row and os.path.basename(row[0]) != target_leaf
        ]

    with open(output_filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, **get_csv_writer_options())
        writer.writerow(header)
        writer.writerows(other_rows)
        writer.writerows(rows)

    return len(rows)


def do_video(
    root, 
    filepath: str,
    debug: bool = False,
    start_paused: bool = False,
    enable_fit_image_to_window: bool = True,
    use_pyav: bool = False,
    rotate_video: bool = False
):
    annotations = Annotations()
    # Dances finished via 'n' within this session - not written to disk yet,
    # all of them get saved together as separate rows when the video closes.
    saved_dances: List[Annotations] = []
    # One linear undo stack spanning the whole session: marker placements
    # AND 'n' dance-boundaries are pushed here in true chronological order,
    # so 'x' always undoes whatever happened most recently, including 'n'
    # itself. Entries: ("waggle_start" | "thorax_position", frame) or
    # ("new_dance", None).
    session_action_history: List[Tuple[str, Optional[int]]] = []
    old_annotations_list = Annotations.load(filepath)
    # Index into (old_annotations_list + saved_dances) of the past dance
    # currently "focused" via '['/']' - that dance shows all its markers too
    # (same as the always-shown active one), for reviewing one dance at a
    # time without needing to scrub into every run's own frame range.
    # None = no extra dance focused.
    focused_dance_index: Optional[int] = None
    # Toggled with 'm': while True, every dance's runs are shown at once,
    # color-coded per dance, and a left-click drag picks up the run nearest
    # the press and reassigns it to whichever dance owns the marker it's
    # dropped on (or a brand-new dance, if dropped on empty space) - a way
    # to fix a run placed into the wrong dance without re-annotating it.
    reassign_mode = False
    # Set for the duration of a reassign-mode drag; None otherwise.
    # Keys: source_dance, dance_number, start_position, end_position,
    # press_pos (screen xy at pickup), screen_pos (current screen xy).
    dragged_run = None

    cap = open_video_capture(filepath)
    cap.set(cv.CAP_PROP_HW_ACCELERATION, cv.VIDEO_ACCELERATION_ANY)

    total_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    video_reader_fps = cap.get(cv.CAP_PROP_FPS)
    video_fps = video_reader_fps if video_reader_fps > 0 else 60

    if use_pyav and av is not None:
        print("Using PyAV library for decoding.")
        cap.release()
        cap = PyavVideoCapture(filepath, total_frames, video_reader_fps)
    elif use_pyav:
        print("PyAV library not found, using OpenCV.")

    capture_cache = VideoCaptureCache(cap, video_fps=video_fps)
    original_frame_image = None
    current_frame = capture_cache.get_current_frame()

    is_in_pause_mode = start_paused
    is_in_draw_vector_mode = False
    hide_past_annotations = False

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Get the video's dimensions
    video_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    if rotate_video:  #then switch the width and height
        print('\n\n\nrotating\n\n\n')
        video_width, video_height = video_height, video_width    
    video_size = (video_width, video_height)
    aspect_ratio = video_width / video_height

     # Create a Toplevel window for the video player
    video_window = tk.Toplevel()

    def update_window_title():
        n_past = len(old_annotations_list) + len(saved_dances)
        focus_label = (
            "none" if focused_dance_index is None
            else f"{focused_dance_index + 1} of {n_past}"
        )
        title = (
            f"Video Annotation Tool - {os.path.basename(filepath)} - "
            f"{video_fps:.2f} FPS - Focused dance: {focus_label}"
        )
        if reassign_mode:
            title += " - REASSIGN MODE"
        video_window.title(title)

    update_window_title()
    video_window.configure(bg=BG)
    # video_window.resizable(False,False)  # disable default resizing in order to keep the aspect

    # Create the widgets first
    speed_scale = BlenderSlider(
        video_window,
        from_=0,
        to=int(3 * video_fps),
        label="Speed (FPS)",
        length=600,
        initial=int(video_fps),  # Set initial speed to video FPS
    )
    speed_scale.pack()


    # Timeline strip, Blender-dopesheet style: diamond markers for every
    # placed waggle_start (orange) and thorax/end (blue) annotation, a frame
    # ruler along the bottom, and a flagged playhead - all clickable to jump
    # straight there.
    timeline_canvas = tk.Canvas(video_window, height=40 + LABEL_HEIGHT, bg=BG_PANEL, highlightthickness=0)
    timeline_canvas.pack(fill=tk.X, padx=4, pady=(0, 2))

    def timeline_frame_to_x(frame, canvas_width):
        span = max(int(total_frames) - 1, 1)
        return int((frame / span) * (canvas_width - 4)) + 2

    def timeline_x_to_frame(x, canvas_width):
        frac = (x - 2) / max(canvas_width - 4, 1)
        frac = min(max(frac, 0.0), 1.0)
        return int(round(frac * (int(total_frames) - 1)))

    def redraw_timeline():
        timeline_canvas.delete("all")
        w = timeline_canvas.winfo_width() or 800
        h = timeline_canvas.winfo_height() or (40 + LABEL_HEIGHT)
        marker_y = LABEL_HEIGHT + (h - LABEL_HEIGHT - RULER_HEIGHT) / 2
        draw_frame_ruler(timeline_canvas, w, h, int(total_frames) - 1, timeline_frame_to_x)
        # Include dances already saved from a previous session
        # (old_annotations_list), plus any queued via 'n' this session, not
        # just the currently-active one, so no dance's markers disappear.
        for dance_number, dance in enumerate(old_annotations_list + saved_dances + [annotations], start=1):
            # Pair starts with thorax/end positions the same way
            # draw_bee_positions does, so the run numbers in these labels
            # match the D{n}-R{m} labels drawn on the video frame itself.
            paired_runs, _, _ = pair_runs(dance)
            run_number_by_frame = {}
            for run_number, (start_position, end_position) in enumerate(paired_runs, start=1):
                run_number_by_frame[start_position.frame] = run_number
                run_number_by_frame[end_position.frame] = run_number

            # In reassign mode, a translucent band in the dance's color spans
            # the full timeline height behind each of its markers, so dance
            # membership is visible on the timeline too - the diamonds
            # themselves keep their usual orange/blue start/end coloring.
            dance_band_color = dance_color_hex(dance_number) if reassign_mode else None

            def draw_dance_band(x):
                timeline_canvas.create_rectangle(
                    x - MARKER_SIZE - 2, 0, x + MARKER_SIZE + 2, h,
                    fill=dance_band_color, outline="", stipple="gray50",
                )

            for position in dance.waggle_starts:
                x = timeline_frame_to_x(position.frame, w)
                if dance_band_color is not None:
                    draw_dance_band(x)
                draw_diamond_marker(timeline_canvas, x, marker_y, ACCENT)
                if not hide_past_annotations and position.frame in run_number_by_frame:
                    timeline_canvas.create_text(
                        x, marker_y - MARKER_SIZE - 2,
                        text=f"D{dance_number}-R{run_number_by_frame[position.frame]}",
                        fill=FG, anchor="s", font=("TkDefaultFont", 7, "bold"),
                    )
            for position in dance.raw_thorax_positions:
                x = timeline_frame_to_x(position.frame, w)
                if dance_band_color is not None:
                    draw_dance_band(x)
                draw_diamond_marker(timeline_canvas, x, marker_y, ACCENT_BLUE)
                if not hide_past_annotations and position.frame in run_number_by_frame:
                    timeline_canvas.create_text(
                        x, marker_y - MARKER_SIZE - 2,
                        text=f"D{dance_number}-R{run_number_by_frame[position.frame]}",
                        fill=FG, anchor="s", font=("TkDefaultFont", 7, "bold"),
                    )
        px = timeline_frame_to_x(current_frame, w)
        draw_playhead(timeline_canvas, px, LABEL_HEIGHT, h - RULER_HEIGHT, frame=current_frame)

    def on_timeline_seek(event):
        w = timeline_canvas.winfo_width()
        move_frame_count(offset=0, target_frame=timeline_x_to_frame(event.x, w))

    timeline_canvas.bind("<Button-1>", on_timeline_seek)
    timeline_canvas.bind("<B1-Motion>", on_timeline_seek)  # click-and-drag to scrub

    # Create the video panel
    video_panel = tk.Label(video_window, bg="black")
    video_panel.pack(expand=True, fill='both')
    # video_panel.pack()
    
    # Force Tkinter to calculate widget sizes
    video_window.update_idletasks()

    # Now retrieve the heights of the widgets
    speed_scale_height = speed_scale.winfo_height()
    timeline_height = timeline_canvas.winfo_height()

    # Initialize the FramePostprocessingPipeline with video size and screen size
    available_height = screen_height - speed_scale_height + timeline_height
    frame_postprocessing_pipeline = FramePostprocessingPipeline(video_size=video_size, screen_size=(screen_width, available_height))

    # Set the initial size of the video window using target_size
    total_window_height = frame_postprocessing_pipeline.target_size[1] + speed_scale_height + timeline_height
    total_window_width = frame_postprocessing_pipeline.target_size[0]
    video_window.geometry(f"{total_window_width}x{total_window_height}")

    # After video capture initialization, ensure the cropping step is initialized
    if frame_postprocessing_pipeline.steps[FramePostprocessingPipeline.CROP] is None:
        # Default mouse position to center of the video size
        default_mouse_position = (
            frame_postprocessing_pipeline.target_size[0] // 2,
            frame_postprocessing_pipeline.target_size[1] // 2,
        )
        placeholder_frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        frame_postprocessing_pipeline.select_next_cropping(
            frame=placeholder_frame,  # The first frame of the video
            mouse_position=default_mouse_position
        )    

    ##### Functions and methods for input handling

    # Variables for mouse position and frame
    last_mouse_position = (0, 0)
    last_raw_frame = None
    
   # Define cleanup when the video window is closed
    def on_video_window_close():
        # Release resources
        if hasattr(capture_cache.capture, 'release'):
            capture_cache.capture.release()

        # Rewrite every dance for this video - ones already saved from a
        # previous session (possibly edited via reassign mode), ones queued
        # via 'n' this session, and whatever's still active now.
        all_dances = old_annotations_list + saved_dances
        if not annotations.is_empty():
            all_dances = all_dances + [annotations]

        n_saved = save_all_dances_for_video(all_dances, filepath)
        if n_saved:
            print(f"Saved {n_saved} dance(s) to {get_output_filename(filepath)}")
        else:
            print("No annotations to save.")

        # Destroy the video window
        video_window.destroy()


    # Key and mouse event handlers
    def on_key_event(event):
        nonlocal is_in_pause_mode, is_in_draw_vector_mode
        nonlocal current_frame, hide_past_annotations, original_frame_image
        nonlocal annotations, focused_dance_index
        nonlocal reassign_mode, dragged_run
        key = event.keysym.lower()
        nframes = calc_frames_to_move_tk(event, video_fps, debug)

        if key == "q":
            # Exit video
            on_video_window_close()
        elif key == "n":
            # Start a fresh dance, without closing the video or losing
            # playback position - for videos with multiple separate dances.
            # Nothing is written to disk here; the current dance is just
            # queued and all queued dances get saved together on quit.
            if not annotations.is_empty():
                saved_dances.append(annotations)
                annotations = Annotations()
                session_action_history.append(("new_dance", None))
        elif key == "space":
            # Pause/unpause
            is_in_pause_mode = not is_in_pause_mode
        elif key == "r":
            # Restart video and clear all annotations, including any
            # dances already queued via 'n' this session.
            capture_cache.set_current_frame(0)
            annotations.clear()
            saved_dances.clear()
            session_action_history.clear()
            original_frame_image = None
            dragged_run = None
        elif key == "a":
            move_frame_count(-nframes)
        elif key == "d":
            move_frame_count(+nframes)
        elif key == "w":
            move_frame_count(+int(video_fps))
        elif key == "s":
            move_frame_count(-int(video_fps))
        elif key == "1":
            move_frame_count(-5 * int(video_fps))
        elif key == "3":
            move_frame_count(+5 * int(video_fps))
        elif key == "plus" or key == "equal":  # 'Equal' key on some keyboards
            speed_scale.set(min(speed_scale.get() + 10, int(3 * video_fps)))
        elif key == "minus":
            speed_scale.set(max(speed_scale.get() - 10, 0))
        elif key == "h":
            hide_past_annotations = not hide_past_annotations
        elif key in ("right", "left"):
            # Cycle which past dance is "focused" (shown fully, like the
            # active one always is) - lets you review one dance at a time
            # without scrubbing into every run's own frame range. Cycles
            # through None (no extra focus) and every past dance.
            n_past = len(old_annotations_list) + len(saved_dances)
            if n_past > 0:
                step = 1 if key == "right" else -1
                if focused_dance_index is None:
                    focused_dance_index = 0 if step > 0 else n_past - 1
                else:
                    next_index = focused_dance_index + step
                    focused_dance_index = next_index if 0 <= next_index < n_past else None
                update_window_title()
        elif key == "m":
            # Toggle reassign mode: shows every dance's runs at once,
            # color-coded, so a run placed into the wrong dance can be
            # drag-and-dropped onto another dance's markers (or empty space,
            # for a brand-new dance) instead of being re-annotated.
            reassign_mode = not reassign_mode
            dragged_run = None
            update_window_title()
        elif key == "c":
            if last_raw_frame is not None:
                frame_postprocessing_pipeline.select_next_contrast_postprocessing(frame=last_raw_frame)
        elif key == "v":
            if last_raw_frame is not None:
                frame_postprocessing_pipeline.select_next_cropping(
                    frame=last_raw_frame, mouse_position=last_mouse_position
                )
        elif key == "x":
            # One linear undo spanning the whole session: pops whatever
            # happened most recently - a marker placement, or an 'n'
            # dance-boundary - and reverses exactly that, regardless of
            # where the playhead is. Undoing 'n' just swaps back to the
            # previous dance; by the time it's undo-able, every marker
            # placed after it has already been popped off first, in order.
            if session_action_history:
                action_kind, action_frame = session_action_history.pop()
                if action_kind == "new_dance":
                    if saved_dances:
                        annotations = saved_dances.pop()
                elif action_kind == "waggle_start":
                    idx = Annotations.get_annotation_index_for_frame(
                        annotations.waggle_starts, action_frame
                    )
                    if idx is not None:
                        del annotations.waggle_starts[idx]
                elif action_kind == "thorax_position":
                    idx = Annotations.get_annotation_index_for_frame(
                        annotations.raw_thorax_positions, action_frame
                    )
                    if idx is not None:
                        del annotations.raw_thorax_positions[idx]
        elif key == "backspace":
            annotations.delete_annotations_on_frame(current_frame)
            # Drop any undo-stack entries this just made stale, so 'x' never
            # tries to "undo" a marker that's already gone.
            session_action_history[:] = [
                entry for entry in session_action_history
                if not (
                    entry[0] in ("waggle_start", "thorax_position")
                    and entry[1] == current_frame
                )
            ]
        elif key == "f":
            max_frame_new = annotations.get_maximum_annotated_frame_index()
            max_frames = []
            for dance in old_annotations_list + saved_dances:
                max_frame_old = dance.get_maximum_annotated_frame_index()
                if max_frame_old is not None:
                    max_frames.append(max_frame_old)
            if max_frame_new is not None:
                max_frames.append(max_frame_new)
            if max_frames:
                max_frame = max(max_frames)
                move_frame_count(offset=0, target_frame=max_frame)

    REASSIGN_HIT_RADIUS = 22  # screen px - how close a click must land to a marker
    REASSIGN_MIN_DRAG_DISTANCE = 8  # screen px - below this, treat as a stray click, not a drag

    def all_dances_numbered():
        # Same 1-based numbering as the D{n}-R{m} labels: previously-saved
        # dances, then ones queued via 'n' this session, then the active one.
        return list(enumerate(old_annotations_list + saved_dances + [annotations], start=1))

    def marker_to_screen(position):
        x, y = position.x, position.y
        if frame_postprocessing_pipeline is not None:
            x, y = frame_postprocessing_pipeline.transform_coordinates_video_to_screen((x, y))
        return x, y

    def find_nearest_marker(screen_x, screen_y, exclude_dance=None):
        # Returns (dance_number, dance, position) for whichever dance's
        # marker is closest to the given screen point, within
        # REASSIGN_HIT_RADIUS - excluding one dance (e.g. the drag's own
        # source, so you can't "drop" a run back onto its own other markers).
        best = None
        best_dist = REASSIGN_HIT_RADIUS
        for dance_number, dance in all_dances_numbered():
            if dance is exclude_dance:
                continue
            for position in dance.waggle_starts + dance.raw_thorax_positions:
                mx, my = marker_to_screen(position)
                dist = ((mx - screen_x) ** 2 + (my - screen_y) ** 2) ** 0.5
                if dist < best_dist:
                    best = (dance_number, dance, position)
                    best_dist = dist
        return best

    def begin_run_drag(screen_x, screen_y):
        nonlocal dragged_run
        hit = find_nearest_marker(screen_x, screen_y)
        if hit is None:
            return
        dance_number, dance, position = hit
        paired_runs, unpaired_starts, unpaired_thorax = pair_runs(dance)
        start_position = end_position = None
        for s, e in paired_runs:
            if s is position or e is position:
                start_position, end_position = s, e
                break
        if start_position is None and end_position is None:
            if position in unpaired_starts:
                start_position = position
            else:
                end_position = position
        dragged_run = dict(
            source_dance=dance,
            dance_number=dance_number,
            start_position=start_position,
            end_position=end_position,
            press_pos=(screen_x, screen_y),
            screen_pos=(screen_x, screen_y),
        )

    def finish_run_drag(screen_x, screen_y):
        nonlocal dragged_run, saved_dances
        run = dragged_run
        dragged_run = None
        if run is None:
            return
        press_x, press_y = run["press_pos"]
        moved_distance = ((screen_x - press_x) ** 2 + (screen_y - press_y) ** 2) ** 0.5
        if moved_distance < REASSIGN_MIN_DRAG_DISTANCE:
            return  # a stray click, not an actual drag - leave the run where it is

        source_dance = run["source_dance"]
        hit = find_nearest_marker(screen_x, screen_y, exclude_dance=source_dance)
        if hit is not None:
            _, target_dance, _ = hit
        else:
            target_dance = Annotations()
            saved_dances.append(target_dance)

        if run["start_position"] is not None:
            source_dance.waggle_starts.remove(run["start_position"])
            target_dance.waggle_starts.append(run["start_position"])
        if run["end_position"] is not None:
            source_dance.raw_thorax_positions.remove(run["end_position"])
            target_dance.raw_thorax_positions.append(run["end_position"])

    def on_mouse_event(event):
        nonlocal is_in_draw_vector_mode, is_in_pause_mode
        nonlocal last_mouse_position

        # Get offsets and displayed image dimensions
        x_offset = getattr(video_panel, 'x_offset', 0)
        y_offset = getattr(video_panel, 'y_offset', 0)
        displayed_image_width = getattr(video_panel, 'displayed_image_width', video_panel.winfo_width())
        displayed_image_height = getattr(video_panel, 'displayed_image_height', video_panel.winfo_height())

        # Adjust mouse coordinates
        x, y = event.x - x_offset, event.y - y_offset

        # Ensure x and y are within the image bounds
        if x < 0 or y < 0 or x >= displayed_image_width or y >= displayed_image_height:
            # Click was outside the image area
            return

        # Update last mouse position
        last_mouse_position = (x, y)

        if reassign_mode:
            # Reassign mode replaces normal marker placement entirely while
            # active - only a button-1 drag does anything. event.num isn't
            # meaningful on plain <Motion> events, so that case is gated by
            # dragged_run being set (i.e. a drag already in progress) rather
            # than by event.num.
            if event.type == tk.EventType.ButtonPress and event.num == 1:
                begin_run_drag(x, y)
            elif event.type == tk.EventType.Motion and dragged_run is not None:
                dragged_run["screen_pos"] = (x, y)
            elif event.type == tk.EventType.ButtonRelease and event.num == 1:
                finish_run_drag(x, y)
            return

        # Transform screen coordinates to video coordinates
        x_video, y_video = frame_postprocessing_pipeline.transform_coordinates_screen_to_video((x, y))
        # x_video, y_video = frame_postprocessing_pipeline.steps[FramePostprocessingPipeline.CROP].transform_coordinates_screen_to_video((x, y))

        is_right_click = (event.num == 3) if sys.platform != "darwin" else (event.num == 2)  # Adjust for Mac vs Windows/Linux

        # A waggle run must be marked as start (position + direction) first,
        # then end (thorax position + frame), alternating 1:1 with no
        # overlap. This makes each thorax position unambiguously the end
        # marker of the waggle_start immediately preceding it.
        is_awaiting_end_marker = len(annotations.waggle_starts) > len(annotations.raw_thorax_positions)

        if event.num == 1:  # Left-click
            if event.type == tk.EventType.ButtonPress:
                is_new_start = Annotations.get_annotation_index_for_frame(
                    annotations.waggle_starts, current_frame
                ) is None
                if is_new_start and is_awaiting_end_marker:
                    messagebox.showwarning(
                        "End position missing",
                        "Please place the thorax position and frame as the "
                        "end position and frame before starting a new waggle run.",
                    )
                    return
                annotations.update_waggle_start(current_frame, x_video, y_video)
                if is_new_start:
                    session_action_history.append(("waggle_start", current_frame))
                is_in_draw_vector_mode = True
                if not is_in_pause_mode:
                    is_in_pause_mode = True
            elif event.type == tk.EventType.ButtonRelease:
                is_in_draw_vector_mode = False
                annotations.update_waggle_direction(current_frame, x_video, y_video)

        elif is_right_click:  # Right-click
            if event.type == tk.EventType.ButtonPress:
                is_new_thorax = Annotations.get_annotation_index_for_frame(
                    annotations.raw_thorax_positions, current_frame
                ) is None
                if is_new_thorax and not is_awaiting_end_marker:
                    messagebox.showwarning(
                        "Start position missing",
                        "Please place the waggle start position and direction first.",
                    )
                    return
                annotations.update_thorax_position(current_frame, x_video, y_video)
                if is_new_thorax:
                    session_action_history.append(("thorax_position", current_frame))

        elif event.type == tk.EventType.Motion:
            last_mouse_position = (x_video, y_video)
            if is_in_draw_vector_mode:
                annotations.update_waggle_direction(current_frame, x_video, y_video)

    def on_mouse_wheel(event):
        nonlocal last_raw_frame, last_mouse_position
        # Determine zoom direction
        if event.num == 4 or event.delta > 0:
            is_zooming_in = True
        elif event.num == 5 or event.delta < 0:
            is_zooming_in = False
        else:
            return  # Unhandled event

        # Adjust mouse coordinates for padding
        x_offset = getattr(video_panel, 'x_offset', 0)
        y_offset = getattr(video_panel, 'y_offset', 0)
        displayed_image_width = getattr(video_panel, 'displayed_image_width', video_panel.winfo_width())
        displayed_image_height = getattr(video_panel, 'displayed_image_height', video_panel.winfo_height())
        x, y = event.x - x_offset, event.y - y_offset

        # Ensure x and y are within the image bounds
        if x < 0 or y < 0 or x >= displayed_image_width or y >= displayed_image_height:
            # Scroll event occurred outside the image area
            return

        # Transform screen coordinates to video coordinates
        x_video, y_video = frame_postprocessing_pipeline.transform_coordinates_screen_to_video((x, y))

        # Update mouse position in the crop step
        frame_postprocessing_pipeline.steps[FramePostprocessingPipeline.CROP].update_mouse_position((x_video, y_video))

        if last_raw_frame is not None:
            direction = 1 if is_zooming_in else -1
            zoom_changed = frame_postprocessing_pipeline.adjust_zoom(direction)
            if zoom_changed:
                current_zoom = frame_postprocessing_pipeline.steps[FramePostprocessingPipeline.CROP].zoom_factor
                print(f"Zoom adjusted to {current_zoom:.2f}x")
            else:
                print("Zoom level unchanged.")

    def calc_frames_to_move_tk(event, video_fps, debug=False):
        # Adjust based on event.state for modifiers
        shift = (event.state & 0x0001) != 0  # Shift key
        ctrl = (event.state & 0x0004) != 0   # Control key
        alt = (event.state & 0x20000) != 0   # Alt key (may vary by platform)
        nframes = 1
        if shift:
            nframes = int(video_fps * 1)
        elif ctrl:
            nframes = int(video_fps * 5)
        elif alt:
            nframes = 5
        return nframes

    def move_frame_count(offset: int, target_frame: Optional[int] = None):
        nonlocal current_frame, original_frame_image

        if target_frame is None:
            target_frame = current_frame + offset

        current_frame = min(max(int(target_frame), 0), int(total_frames) - 1)
        if is_in_pause_mode or (current_frame - capture_cache.get_current_frame() > 1):
            capture_cache.set_current_frame(current_frame)

        # Clear current raw frame so we fetch a new one even if in pause mode.
        original_frame_image = None

        # Keep the timeline's playhead in sync immediately, rather than
        # waiting for the next render tick.
        redraw_timeline()

    # Bind events
    video_panel.bind("<ButtonPress-1>", on_mouse_event)  # Left-click
    video_panel.bind("<ButtonRelease-1>", on_mouse_event)  # Left-click release
    video_panel.bind("<ButtonPress-2>", on_mouse_event)  # Right-click (Mac)
    video_panel.bind("<ButtonPress-3>", on_mouse_event)  # Right-click (Windows/Linux)
    video_panel.bind("<Motion>", on_mouse_event)  # Mouse movement
    video_panel.bind("<MouseWheel>", on_mouse_wheel)  # Windows and MacOS
    video_panel.bind("<Button-4>", on_mouse_wheel)    # Linux scroll up
    video_panel.bind("<Button-5>", on_mouse_wheel)    # Linux scroll down

    # Bind the key event handler to the video window
    video_window.bind("<Key>", on_key_event)

    def on_resize(event):
        # Get new window dimensions
        window_width = event.width
        window_height = event.height

        # Get the height of the speed scale and timeline to calculate available space for the video panel
        speed_scale_height = speed_scale.winfo_height() or 0
        timeline_height = timeline_canvas.winfo_height() or 0
        available_height = window_height - (speed_scale_height + timeline_height)
        available_width = window_width  # The width available is the window's width

        # Ensure positive available dimensions
        available_height = max(available_height, 1)
        available_width = max(available_width, 1)

        # Calculate the video's aspect ratio
        crop_step = frame_postprocessing_pipeline.steps[FramePostprocessingPipeline.CROP]
        if crop_step is None:
            return  # Crop step not initialized yet

        aspect_ratio = crop_step.aspect_ratio

        # Calculate new dimensions while preserving aspect ratio
        if available_width / available_height > aspect_ratio:
            # Available space is wider than the video aspect ratio
            new_height = available_height
            new_width = int(new_height * aspect_ratio)
        else:
            # Available space is taller than the video aspect ratio
            new_width = available_width
            new_height = int(new_width / aspect_ratio)

        # Ensure new_width and new_height are positive
        new_width = max(new_width, 1)
        new_height = max(new_height, 1)

        # Update the target size in the pipeline
        crop_step.target_size = (new_width, new_height)
        crop_step.update_scaling_factors()     

    # Bind the window resize event
    video_window.bind("<Configure>", on_resize)    

    # Ensure focus is on the video window for key events
    video_window.focus_force()    

    # @profile  # with this in place, run:  kernprof -l -v main.py -p /Users/jacob/Desktop/waggle_label
    def update_frame():
        nonlocal current_frame, original_frame_image, last_raw_frame, is_in_pause_mode

        # Get the speed from the speed_scale
        current_speed = speed_scale.get()

        # Only set is_in_pause_mode to True if current_speed is 0
        if current_speed == 0:
            is_in_pause_mode = True
        # Do not set is_in_pause_mode to False here to respect user pause

        if not is_in_pause_mode or original_frame_image is None:
            # Read the current frame
            current_frame = capture_cache.get_current_frame()
            has_valid_frame, original_frame_image = capture_cache.read()
            if rotate_video:  # rotate here already, so that all steps use the rotated image
                original_frame_image = cv.rotate(original_frame_image, cv.ROTATE_90_CLOCKWISE)
            if original_frame_image is not None:
                last_raw_frame = original_frame_image

                # **Initialize the cropping step after the first frame is loaded**
                if frame_postprocessing_pipeline.steps[FramePostprocessingPipeline.CROP] is None:
                    # Default mouse position to center of the target size
                    default_mouse_position = (
                        frame_postprocessing_pipeline.target_size[0] // 2,
                        frame_postprocessing_pipeline.target_size[1] // 2,
                    )
                    frame_postprocessing_pipeline.select_next_cropping(
                        frame=original_frame_image,
                        mouse_position=default_mouse_position
                    )
        else:
            has_valid_frame = True

        if not has_valid_frame:
            move_frame_count(-1)
            is_in_pause_mode = True
            # Still reschedule, otherwise the update loop dies here and the
            # display never refreshes again, even if the frame slider is moved.
            delay = int(1000 / (speed_scale.get() if speed_scale.get() > 0 else 1))
            video_window.after(delay, update_frame)
            return

        # frame = original_frame_image.copy()
        frame = original_frame_image
        frame = frame_postprocessing_pipeline.process(frame)

        frame = draw_template(
            frame,
            current_time=calculate_time(current_frame, video_fps),
        )
        past_dances = old_annotations_list + saved_dances
        active_dance_number = len(past_dances) + 1
        frame = draw_bee_positions(
            frame,
            annotations,
            current_frame=current_frame,
            hide_past_annotations=hide_past_annotations,
            frame_postprocessing_pipeline=frame_postprocessing_pipeline,
            show_all=True,  # always show every run of the dance being actively annotated
            dance_number=active_dance_number,
            show_labels=not hide_past_annotations or reassign_mode,
        )
        if not hide_past_annotations or reassign_mode:
            # Dances already queued via 'n' this session render the same way
            # as previously-saved ones, so they stay visible going forward.
            # Reassign mode shows every run of every dance regardless of
            # frame range or focus, so every drop target is visible at once.
            for dance_index, old_annotations in enumerate(past_dances):
                frame = draw_bee_positions(
                    frame,
                    old_annotations,
                    current_frame=current_frame,
                    dance_number=dance_index + 1,
                    show_labels=True,
                    show_all=(reassign_mode or dance_index == focused_dance_index),
                    frame_postprocessing_pipeline=frame_postprocessing_pipeline
                )

        if reassign_mode:
            frame = draw_reassign_overlay(frame, all_dances_numbered(), frame_postprocessing_pipeline)
            if dragged_run is not None:
                gx, gy = dragged_run["screen_pos"]
                drag_color = dance_color_bgr(dragged_run["dance_number"])
                cv.circle(frame, (int(gx), int(gy)), 12, drag_color, -1)
                cv.circle(frame, (int(gx), int(gy)), 12, (255, 255, 255), 2)
            banner = "REASSIGN MODE - drag a run onto another dance's circle (empty space = new dance)"
            cv.putText(frame, banner, (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv.LINE_AA)
            cv.putText(frame, banner, (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

        # Convert the frame to an image that can be displayed in Tkinter
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=image)

        # Update the video panel with the new frame
        video_panel.imgtk = imgtk  # Keep a reference to prevent garbage collection
        video_panel.configure(image=imgtk)

        # Store displayed image dimensions
        displayed_image_width = image.width
        displayed_image_height = image.height

        # Get video_panel dimensions
        panel_width = video_panel.winfo_width()
        panel_height = video_panel.winfo_height()

        # Calculate offsets
        x_offset = (panel_width - displayed_image_width) // 2
        y_offset = (panel_height - displayed_image_height) // 2

        # Store these for use in on_mouse_event
        video_panel.displayed_image_width = displayed_image_width
        video_panel.displayed_image_height = displayed_image_height
        video_panel.x_offset = x_offset
        video_panel.y_offset = y_offset

        # Get the speed from the speed_scale
        current_speed = speed_scale.get()
        if current_speed == 0:
            is_in_pause_mode = True

        redraw_timeline()

        # Schedule the next frame update
        delay = int(1000 / (speed_scale.get() if speed_scale.get() > 0 else 1))
        video_window.after(delay, update_frame)

    # Start the frame update loop
    update_frame()

    # Bind the close event
    video_window.protocol("WM_DELETE_WINDOW", on_video_window_close)

    # Wait for the video window to be closed
    video_window.wait_window()
