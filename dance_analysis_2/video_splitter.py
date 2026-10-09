# video_splitter.py
#
# Standalone tool to mark split points on a video while previewing it, then
# export each marked range as its own file via ffmpeg stream-copy. The
# source video is never modified.

import argparse
import glob
import json
import os
import re
import subprocess
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Optional

import cv2 as cv
from PIL import Image, ImageTk

from .utils import open_video_capture

VIDEO_EXTENSIONS = (
    ".mp4", ".avi", ".h264", ".mov", ".mkv",
    ".mpeg", ".mpg", ".wmv", ".flv", ".m4v",
    ".3gp", ".3g2", ".mts", ".m2ts",
)


def get_keyframe_timestamps(filepath: str) -> List[float]:
    """Query keyframe (IDR) timestamps in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-skip_frame", "nokey",
        "-select_streams", "v:0",
        "-show_entries", "frame=pts_time",
        "-of", "csv=print_section=0",
        filepath,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return sorted(
            float(line.strip().rstrip(","))
            for line in result.stdout.splitlines()
            if line.strip()
        )
    except Exception as e:
        print(f"Could not read keyframe timestamps: {e}")
        return []


def get_markers_filepath(filepath: str) -> str:
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    return os.path.join(os.path.dirname(filepath), f"{base_name}_split_markers.json")


def load_saved_markers(filepath: str) -> List[int]:
    """Loads previously-placed split points for this video, if any were saved."""
    markers_path = get_markers_filepath(filepath)
    if not os.path.exists(markers_path):
        return []
    try:
        with open(markers_path, "r") as f:
            data = json.load(f)
        return sorted(int(f) for f in data.get("split_frames", []))
    except Exception as e:
        print(f"Could not read saved split markers ({markers_path}): {e}")
        return []


def save_markers(filepath: str, split_frames: List[int]) -> None:
    """Persists the current split points so they survive closing/reopening the tool."""
    markers_path = get_markers_filepath(filepath)
    try:
        with open(markers_path, "w") as f:
            json.dump(
                {"video_name": os.path.basename(filepath), "split_frames": sorted(split_frames)},
                f,
            )
    except Exception as e:
        print(f"Could not save split markers ({markers_path}): {e}")


def get_next_numbered_output(out_dir: str, base_name: str, ext: str, tag: str = "seg") -> int:
    """
    Finds the highest existing <tag><N> for this video in out_dir and returns
    the next free number, so re-running an export never overwrites earlier
    output. Used for both dance-run segments ('seg') and coarse pre-split
    chunks ('chunk'), each numbered independently.
    """
    pattern = os.path.join(out_dir, f"{glob.escape(base_name)}-*-{tag}*{glob.escape(ext)}")
    number_re = re.compile(re.escape(base_name) + rf"-.*-{tag}(\d+)" + re.escape(ext) + r"$")
    highest = 0
    for candidate in glob.glob(pattern):
        match = number_re.match(os.path.basename(candidate))
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def nearest_keyframe_at_or_before(timestamps: List[float], t: float) -> Optional[float]:
    candidates = [kf for kf in timestamps if kf <= t]
    if candidates:
        return max(candidates)
    return timestamps[0] if timestamps else None


def _split_hms(seconds: float):
    seconds = max(seconds, 0.0)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        millis = 0
        secs += 1
        if secs == 60:
            secs = 0
            minutes += 1
            if minutes == 60:
                minutes = 0
                hours += 1
    return hours, minutes, secs, millis


def format_timestamp(seconds: float) -> str:
    # Matches the existing "<basename>-HH.MM.SS.mmm-HH.MM.SS.mmm-segN.ext" convention.
    hours, minutes, secs, millis = _split_hms(seconds)
    return f"{hours:02d}.{minutes:02d}.{secs:02d}.{millis:03d}"


def format_display_time(seconds: float) -> str:
    # On-screen only (never used in filenames): HH:MM:SS.mmm, independent of fps.
    hours, minutes, secs, millis = _split_hms(seconds)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def auto_split_video(filepath: str, chunk_minutes: float) -> List[str]:
    """
    Non-interactively cuts a video into fixed-length chunks (~chunk_minutes
    each, last one shorter) via ffmpeg stream-copy - no GUI, no keyframe
    scan, no marker placement. Meant as a cheap pre-processing pass on very
    long/large recordings, so the interactive splitter only ever has to load
    a manageable chunk instead of the entire multi-hour source.

    Output goes to <basename>_chunks/, named <basename>-chunkN-<start>-<end>.<ext>
    - deliberately a different folder/tag than the interactive tool's
    <basename>_segs/<...>-segN.<ext>, so coarse pre-split chunks are never
    confused with actual marked dance-run segments.
    """
    cap = open_video_capture(filepath)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {filepath}")
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv.CAP_PROP_FPS)
    cap.release()
    if not video_fps or video_fps <= 0:
        video_fps = 30.0
    total_duration = total_frames / video_fps

    chunk_seconds = chunk_minutes * 60
    n_chunks = max(int(-(-total_duration // chunk_seconds)), 1)  # ceil div

    base_name = os.path.splitext(os.path.basename(filepath))[0]
    ext = os.path.splitext(filepath)[1]
    out_dir = os.path.join(os.path.dirname(filepath), f"{base_name}_chunks")
    os.makedirs(out_dir, exist_ok=True)
    start_chunk_number = get_next_numbered_output(out_dir, base_name, ext, tag="chunk")

    print(
        f"Splitting {os.path.basename(filepath)} "
        f"({format_display_time(total_duration)} total) into {n_chunks} "
        f"chunk(s) of ~{chunk_minutes:.0f} min each -> {out_dir}"
    )

    exported_paths = []
    for i in range(n_chunks):
        chunk_number = start_chunk_number + i
        start_t = i * chunk_seconds
        end_t = min((i + 1) * chunk_seconds, total_duration)
        duration = end_t - start_t
        out_name = (
            f"{base_name}-chunk{chunk_number}-"
            f"{format_timestamp(start_t)}-{format_timestamp(end_t)}{ext}"
        )
        out_path = os.path.join(out_dir, out_name)

        if os.path.exists(out_path):
            raise RuntimeError(f"Refusing to overwrite existing file: {out_path}")

        print(f"  [{i + 1}/{n_chunks}] {out_name} ...")
        cmd = [
            "ffmpeg", "-n",
            "-ss", str(start_t),
            "-i", filepath,
            "-t", str(duration),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed on chunk {i + 1}:\n{result.stderr[-2000:]}")
        exported_paths.append(out_path)

    print(f"Done. Wrote {len(exported_paths)} chunk(s) to {out_dir}")
    return exported_paths


def do_split_tool(root: tk.Tk, filepath: str):
    cap = open_video_capture(filepath)
    if not cap.isOpened():
        messagebox.showerror("Could not open video", f"Failed to open:\n{filepath}")
        return

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        video_fps = 30.0
    total_duration_s = total_frames / video_fps

    saved_markers = load_saved_markers(filepath)
    state = dict(
        current_frame=0,
        split_frames=list(saved_markers),  # sorted list of frame indices where a cut is marked
        marker_insertion_order=list(saved_markers),  # same frames, in the order they were added (for undo-last)
        keyframe_timestamps=[],
        keyframes_ready=False,
        is_playing=False,
        last_read_frame=-1,  # tracks sequential reads so we only reseek on a real jump
        zoom=1.0,  # scales the whole displayed frame up/down - no cropping
    )

    win = tk.Toplevel(root)
    win.title(f"Video Splitter - {os.path.basename(filepath)} - {video_fps:.2f} FPS")

    video_panel = tk.Label(win, bg="black")
    video_panel.pack(expand=True, fill="both")

    # The scale still operates on frame indices internally (needed for exact
    # per-frame seeking/export), but the displayed value is time-based, since
    # a raw frame number isn't meaningfully comparable across videos loaded
    # at different fps.
    frame_scale = tk.Scale(
        win, from_=0, to=max(total_frames - 1, 0), orient=tk.HORIZONTAL,
        label="Position", length=900, showvalue=0,
    )
    frame_scale.pack(fill=tk.X, padx=10)

    time_label = tk.Label(
        win,
        text=f"{format_display_time(0)} / {format_display_time(total_duration_s)}",
        font=("TkDefaultFont", 12, "bold"),
    )
    time_label.pack(fill=tk.X, padx=10)

    status_label = tk.Label(win, text="Scanning keyframes for accurate-cut preview...", fg="gray")
    status_label.pack(fill=tk.X, padx=10)

    play_state_label = tk.Label(win, text="Paused (space to play)", fg="gray")
    play_state_label.pack(fill=tk.X, padx=10)

    zoom_label = tk.Label(win, text="Zoom: 1.0x (scroll over video to zoom)", fg="gray")
    zoom_label.pack(fill=tk.X, padx=10)

    timeline_canvas = tk.Canvas(win, height=50, bg="#1a1a1a", highlightthickness=0)
    timeline_canvas.pack(fill=tk.X, padx=10, pady=(4, 10))

    controls_frame = tk.Frame(win)
    controls_frame.pack(fill=tk.X, padx=10, pady=(0, 6))

    marker_listbox = tk.Listbox(win, height=6)
    marker_listbox.pack(fill=tk.X, padx=10, pady=(0, 10))

    # ---- helpers -----------------------------------------------------

    def frame_to_x(frame: int, canvas_width: int) -> int:
        span = max(total_frames - 1, 1)
        return int((frame / span) * (canvas_width - 4)) + 2

    def x_to_frame(x: int, canvas_width: int) -> int:
        frac = (x - 2) / max(canvas_width - 4, 1)
        frac = min(max(frac, 0.0), 1.0)
        return int(round(frac * (total_frames - 1)))

    def redraw_timeline():
        timeline_canvas.delete("all")
        w = timeline_canvas.winfo_width() or 900
        h = timeline_canvas.winfo_height() or 50

        for frame in state["split_frames"]:
            mx = frame_to_x(frame, w)
            timeline_canvas.create_line(mx, 0, mx, h, fill="#ff8800", width=1, dash=(3, 2))
            if state["keyframes_ready"]:
                t = frame / video_fps
                snapped_t = nearest_keyframe_at_or_before(state["keyframe_timestamps"], t)
                if snapped_t is not None:
                    snapped_frame = int(round(snapped_t * video_fps))
                    sx = frame_to_x(snapped_frame, w)
                    timeline_canvas.create_line(sx, 0, sx, h, fill="#ff3333", width=2)

        px = frame_to_x(state["current_frame"], w)
        timeline_canvas.create_line(px, 0, px, h, fill="white", width=2)

    def show_frame():
        # A manual seek (cap.set) is expensive: it has to locate the nearest
        # keyframe and decode forward from there. Only do it on a real jump
        # (slider drag, timeline click) - sequential autoplay just reads the
        # next frame directly, which is what keeps it near real-time.
        if state["current_frame"] != state["last_read_frame"] + 1:
            cap.set(cv.CAP_PROP_POS_FRAMES, state["current_frame"])
        ret, img = cap.read()
        state["last_read_frame"] = state["current_frame"]
        current_t = state["current_frame"] / video_fps
        time_label.config(
            text=f"{format_display_time(current_t)} / {format_display_time(total_duration_s)}"
        )
        if not ret:
            return

        # Zoom just scales the whole frame up/down for display - no cropping,
        # so the full picture is always visible, just bigger or smaller.
        h, w = img.shape[:2]
        max_w = int(1000 * state["zoom"])
        scale = max_w / w
        img = cv.resize(img, (int(w * scale), int(h * scale)))

        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        video_panel.imgtk = photo  # keep a reference
        video_panel.configure(image=photo)
        video_panel.displayed_image_width = img.shape[1]
        video_panel.displayed_image_height = img.shape[0]
        zoom_label.config(text=f"Zoom: {state['zoom']:.1f}x (scroll over video to zoom)")
        redraw_timeline()

    def seek_to_frame(frame: int):
        frame = min(max(int(frame), 0), max(total_frames - 1, 0))
        state["current_frame"] = frame
        # Detach the command callback while we set the scale programmatically.
        # Tk's Scale invokes 'command' asynchronously (queued, not inline), so
        # a flag reset right after .set() doesn't reliably suppress it - fully
        # unbinding does, since there's then nothing registered to fire.
        frame_scale.config(command="")
        frame_scale.set(frame)
        frame_scale.config(command=on_frame_scale_change)
        show_frame()

    def on_frame_scale_change(value):
        new_frame = int(float(value))
        if state["is_playing"]:
            toggle_play()  # manual scrub pauses playback
        if new_frame != state["current_frame"]:
            seek_to_frame(new_frame)

    frame_scale.config(command=on_frame_scale_change)

    target_frame_interval_ms = 1000 / video_fps

    def advance_playback():
        if not state["is_playing"]:
            return
        next_frame = state["current_frame"] + 1
        if next_frame >= total_frames:
            state["is_playing"] = False
            play_state_label.config(text="Paused (space to play)", fg="gray")
            return
        cycle_start = time.time()
        seek_to_frame(next_frame)
        elapsed_ms = (time.time() - cycle_start) * 1000
        # Subtract the time this cycle's own work took, so per-frame
        # processing overhead doesn't silently halve the playback rate.
        delay = max(int(target_frame_interval_ms - elapsed_ms), 1)
        win.after(delay, advance_playback)

    def toggle_play(event=None):
        state["is_playing"] = not state["is_playing"]
        if state["is_playing"]:
            play_state_label.config(text="Playing (space to pause)", fg="#2e7d32")
            advance_playback()
        else:
            play_state_label.config(text="Paused (space to play)", fg="gray")

    def on_timeline_click(event):
        w = timeline_canvas.winfo_width()
        seek_to_frame(x_to_frame(event.x, w))

    timeline_canvas.bind("<Button-1>", on_timeline_click)

    def on_mouse_wheel(event):
        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            zooming_in = True
        elif getattr(event, "num", None) == 5 or getattr(event, "delta", 0) < 0:
            zooming_in = False
        else:
            return

        new_zoom = state["zoom"] * (1.15 if zooming_in else 1 / 1.15)
        state["zoom"] = min(max(new_zoom, 0.5), 3.0)
        show_frame()

    video_panel.bind("<MouseWheel>", on_mouse_wheel)  # Windows/macOS
    video_panel.bind("<Button-4>", on_mouse_wheel)  # Linux scroll up
    video_panel.bind("<Button-5>", on_mouse_wheel)  # Linux scroll down

    def update_marker_list():
        marker_listbox.delete(0, tk.END)
        for f in state["split_frames"]:
            t = f / video_fps
            marker_listbox.insert(tk.END, f"{format_display_time(t)}  (frame {f})")

    def add_split_point():
        f = state["current_frame"]
        if 0 < f < total_frames - 1 and f not in state["split_frames"]:
            state["split_frames"].append(f)
            state["split_frames"].sort()
            state["marker_insertion_order"].append(f)
            save_markers(filepath, state["split_frames"])
            redraw_timeline()
            update_marker_list()
        # Return focus to the window so a subsequent space-press toggles
        # play/pause instead of re-clicking this button (Tk buttons treat
        # space as "activate" while focused).
        win.focus_set()

    def remove_last_marker():
        if not state["marker_insertion_order"]:
            return
        f = state["marker_insertion_order"].pop()
        state["split_frames"].remove(f)
        save_markers(filepath, state["split_frames"])
        redraw_timeline()
        update_marker_list()
        win.focus_set()

    def remove_selected_marker():
        sel = marker_listbox.curselection()
        if not sel:
            win.focus_set()
            return
        f = state["split_frames"][sel[0]]
        del state["split_frames"][sel[0]]
        state["marker_insertion_order"].remove(f)
        save_markers(filepath, state["split_frames"])
        redraw_timeline()
        update_marker_list()
        win.focus_set()

    def calc_frames_to_step(event) -> int:
        # Same modifier scheme as the annotator, but the plain (no modifier)
        # step is ~0.25s of video rather than a single frame - a bare frame
        # is too fine-grained to be a useful default at ~60fps.
        shift = (event.state & 0x0001) != 0
        ctrl = (event.state & 0x0004) != 0
        alt = (event.state & 0x20000) != 0
        if shift:
            return int(video_fps * 1)
        if ctrl:
            return int(video_fps * 5)
        if alt:
            return 5
        return max(round(video_fps * 0.25), 1)

    def step_frames(offset: int):
        if state["is_playing"]:
            toggle_play()  # stepping while playing pauses first, like a scrub
        seek_to_frame(state["current_frame"] + offset)

    def on_key_event(event):
        key = event.keysym.lower()
        if key == "space":
            toggle_play()
        elif key == "a":
            step_frames(-calc_frames_to_step(event))
        elif key == "d":
            step_frames(calc_frames_to_step(event))
        elif key == "w":
            step_frames(int(video_fps))
        elif key == "s":
            step_frames(-int(video_fps))
        elif key == "1":
            step_frames(-5 * int(video_fps))
        elif key == "3":
            step_frames(5 * int(video_fps))
        elif key == "k":
            add_split_point()
        elif key == "x":
            remove_last_marker()

    win.bind("<Key>", on_key_event)
    win.focus_set()

    def export_segments():
        if not state["split_frames"]:
            messagebox.showinfo(
                "No split points", "Place at least one split point before exporting."
            )
            return

        boundaries = [0] + state["split_frames"] + [total_frames]
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        ext = os.path.splitext(filepath)[1]
        out_dir = os.path.join(os.path.dirname(filepath), f"{base_name}_segs")
        os.makedirs(out_dir, exist_ok=True)
        n_segments = len(boundaries) - 1

        # Continue numbering after whatever seg<N> files already exist in
        # this video's own <base_name>_segs/ folder, so re-running the
        # splitter never overwrites earlier exports, and segments from
        # different source videos never mix together.
        start_seg_number = get_next_numbered_output(out_dir, base_name, ext, tag="seg")

        export_win = tk.Toplevel(win)
        export_win.title("Exporting")
        progress_label = tk.Label(export_win, text="", padx=20, pady=20)
        progress_label.pack()
        export_win.update()

        exported_paths = []
        for i in range(n_segments):
            seg_number = start_seg_number + i
            start_t = boundaries[i] / video_fps
            end_t = boundaries[i + 1] / video_fps
            duration = end_t - start_t
            out_name = (
                f"{base_name}-{format_timestamp(start_t)}-"
                f"{format_timestamp(end_t)}-seg{seg_number}{ext}"
            )
            out_path = os.path.join(out_dir, out_name)

            if os.path.exists(out_path):
                export_win.destroy()
                messagebox.showerror(
                    "Export failed",
                    f"Refusing to overwrite existing file:\n{out_path}",
                )
                return

            progress_label.config(
                text=f"Exporting segment {i + 1}/{n_segments} (seg{seg_number})\n{out_name}"
            )
            export_win.update()

            cmd = [
                "ffmpeg", "-n",
                "-ss", str(start_t),
                "-i", filepath,
                "-t", str(duration),
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                out_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                export_win.destroy()
                messagebox.showerror(
                    "Export failed",
                    f"ffmpeg failed on segment {i + 1}:\n{result.stderr[-2000:]}",
                )
                return
            exported_paths.append(out_path)

        export_win.destroy()

        # Clear markers now that they're baked into real output files -
        # otherwise reopening this video later would reload them and risk
        # re-exporting the same cuts again under new segment numbers.
        state["split_frames"].clear()
        state["marker_insertion_order"].clear()
        markers_path = get_markers_filepath(filepath)
        if os.path.exists(markers_path):
            os.remove(markers_path)
        redraw_timeline()
        update_marker_list()

        messagebox.showinfo(
            "Export complete",
            f"Exported {n_segments} segment(s) to:\n{out_dir}\n\n"
            + "\n".join(os.path.basename(p) for p in exported_paths)
            + "\n\nMarkers cleared (already exported).",
        )

    marker_listbox.bind("<Double-Button-1>", lambda event: remove_selected_marker())

    # A plain tk.Button ignores bg on macOS's native (Aqua) theme - it always
    # renders as the native white/gray button regardless of what's set here,
    # which made white text invisible against it. A styled Label bound to a
    # click does honor bg/fg reliably across platforms.
    export_button = tk.Label(
        controls_frame,
        text="Export segments",
        bg="#2e7d32",
        fg="white",
        padx=10,
        pady=4,
        relief=tk.RAISED,
        cursor="hand2",
    )
    export_button.pack(side=tk.RIGHT)
    export_button.bind("<Button-1>", lambda event: export_segments())

    legend = tk.Label(
        win,
        text="orange dashed = where you clicked   |   red solid = actual cut point (nearest keyframe)",
        fg="gray",
    )
    legend.pack(fill=tk.X, padx=10, pady=(0, 6))

    keys_legend = tk.Label(
        win,
        justify=tk.LEFT,
        anchor="w",
        text=(
            "How to use  —  "
            "space: play/pause   ·   "
            "a/d: ±0.25s (shift ±1s, ctrl ±5s, alt ±5 frames)   ·   "
            "w/s: ±1s   ·   "
            "1/3: ±5s   ·   "
            "k: add split point here   ·   "
            "x: undo last marker   ·   "
            "double-click a marker below to remove it"
        ),
        fg="gray",
    )
    keys_legend.pack(fill=tk.X, padx=10, pady=(0, 6))

    def on_keyframes_ready(timestamps: List[float]):
        if not win.winfo_exists():
            # Window was closed while the background scan was still running.
            return
        state["keyframe_timestamps"] = timestamps
        state["keyframes_ready"] = True
        status_label.config(text=f"Ready ({len(timestamps)} keyframes found).")
        redraw_timeline()

    def scan_keyframes_in_background():
        timestamps = get_keyframe_timestamps(filepath)
        try:
            win.after(0, lambda: on_keyframes_ready(timestamps))
        except tk.TclError:
            # Window was already destroyed; nothing left to update.
            pass

    threading.Thread(target=scan_keyframes_in_background, daemon=True).start()

    def on_close():
        cap.release()
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", on_close)

    update_marker_list()  # show any markers restored from a previous session
    seek_to_frame(0)


def main():
    parser = argparse.ArgumentParser(
        description="Mark split points on a video and export the marked ranges as separate files."
    )
    parser.add_argument(
        "-f", "--file", type=str, default=None, help="path to the video file to split"
    )
    parser.add_argument(
        "--auto-split-minutes",
        type=float,
        default=None,
        help=(
            "Non-interactively pre-chunk the video into fixed ~N-minute pieces "
            "(no GUI) instead of opening the interactive marker tool. Useful "
            "before marking dance runs in a very long/large recording, so the "
            "interactive tool only ever has to load a manageable chunk."
        ),
    )
    args = parser.parse_args()

    filepath = args.file
    if not filepath:
        # A plain Tk() root is enough for a file dialog; the interactive
        # tool creates its own withdrawn root separately below.
        picker_root = tk.Tk()
        picker_root.withdraw()
        filepath = filedialog.askopenfilename(
            title="Select a video to split",
            filetypes=[
                ("Video files", " ".join(f"*{ext}" for ext in VIDEO_EXTENSIONS)),
                ("All files", "*.*"),
            ],
        )
        picker_root.destroy()
    if not filepath:
        print("No file selected. Exiting.")
        return

    if args.auto_split_minutes is not None:
        auto_split_video(filepath, args.auto_split_minutes)
        return

    root = tk.Tk()
    root.withdraw()

    def on_root_close():
        root.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_root_close)

    do_split_tool(root, filepath)
    root.mainloop()


if __name__ == "__main__":
    main()
