# ui.py

import os
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import pandastable
from typing import Callable, Dict, List, Tuple
from .annotations import Annotations
from .utils import get_output_filename

class FileSelectorUI:
    def __init__(self, root, root_path: str, on_filepath_selected: Callable):
        self.root = root  # Use the root passed in
        self.root_path = root_path
        self.on_filepath_selected = on_filepath_selected
        self.index_map: Dict[int, str] = {}
        self.instructions_frame = None
        self.checkboxes: List[Tuple[str, tk.IntVar]] = []
        self.pt = None  # pandastable.Table instance
        self.table_frame = None  # Frame for the table
        self.checkbox_frame = None  # Frame for the checkboxes
        self.selected_video = None  # Holds the selected video filepath

    def collect_files(self):
        # List of common video file extensions
        video_extensions = (
            '.mp4', '.avi', '.h264', '.mov', '.mkv',
            '.mpeg', '.mpg', '.wmv', '.flv', '.m4v',
            '.3gp', '.3g2'
        )

        for root_dir, dirs, files in os.walk(self.root_path):
            for name in files:
                # Check if the file ends with any of the video extensions (case-insensitive)
                if name.lower().endswith(video_extensions):
                    filepath = os.path.join(root_dir, name)
                    yield filepath, name

    def load_old_annotation_infos(self, filepath: str) -> Dict[str, int]:
        annotations_list = Annotations.load(filepath, on_error="silent")

        n_waggle_starts = 0
        n_thorax_points = 0
        max_annotated_frame = 0
        n_dances = 0

        if annotations_list:
            n_waggle_starts = sum(len(a.waggle_starts) for a in annotations_list)
            n_thorax_points = sum(len(a.raw_thorax_positions) for a in annotations_list)
            max_annotated_frame = max(
                int(a.get_maximum_annotated_frame_index() or 0) for a in annotations_list
            )
            n_dances = len(annotations_list)

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

        # Get additional processing kwargs
        processing_kwargs = self.get_additional_processing_kwargs()

        # Store the selected video filepath
        self.selected_video = filepath

        # Hide the UI window instead of destroying it
        self.root.withdraw()

        # Start the video processing in the main thread
        self.on_filepath_selected(filepath, **processing_kwargs)

        # After video processing is done, re-show the UI window
        self.root.deiconify()

        # Optionally, update the table with new annotation data
        annotation_infos = self.load_old_annotation_infos(filepath)
        for key, value in annotation_infos.items():
            raw_table.at[idx, key] = value

        self.pt.redraw()

    def create_instructions_table(self, parent):
        # Create a frame for the instructions table
        self.instructions_frame = tk.Frame(parent)

        # Title for the instructions
        title_label = tk.Label(
            self.instructions_frame,
            text="Instructions and Key Mappings",
            font=('Arial', 14, 'bold')
        )
        title_label.pack(pady=(5, 10))

        # Table Frame
        table_frame = tk.Frame(self.instructions_frame)
        table_frame.pack(fill="x")

        # Headers
        headers = ["Key", "Description"]
        for col_num, header in enumerate(headers):
            label = tk.Label(
                table_frame,
                text=header,
                font=('Arial', 12, 'bold'),
                borderwidth=1,
                relief="solid",
                padx=5,
                pady=5
            )
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
            key_label = tk.Label(
                table_frame,
                text=key,
                font=('Arial', 12),
                borderwidth=1,
                relief="solid",
                padx=5,
                pady=5
            )
            key_label.grid(row=row_num, column=0, sticky="nsew")
            desc_label = tk.Label(
                table_frame,
                text=desc,
                font=('Arial', 12),
                borderwidth=1,
                relief="solid",
                padx=5,
                pady=5
            )
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
        # Use the root window passed in
        self.root.deiconify()  # Show the root window
        self.root.title("Available Videos")
        self.root.geometry("800x1000")
        self.index_map = {}

        table_data = []
        for idx, (filepath, filename) in enumerate(self.collect_files()):
            self.index_map[idx] = filepath

            metadata = dict(index=idx, filename=filename)
            metadata.update(self.load_old_annotation_infos(filepath))

            table_data.append(metadata)

        if not table_data:
            messagebox.showerror("No files found", "No video files available. Stopping.")
            return

        table_df = pd.DataFrame(table_data).sort_values("filename")
        table_df.set_index("index", inplace=True)

        # Create the instructions table and toggle button
        self.create_instructions_table(self.root)  # first create the table object
        toggle_button = tk.Button(
            self.root,
            text="Show/Hide Instructions",
            command=self.toggle_instructions
        )
        toggle_button.pack(pady=10)

        # Create the checkbox frame
        self.checkbox_frame = tk.Frame(self.root)
        self.checkbox_frame.pack(fill="x", expand=True)

        # Create checkboxes for additional options
        self.checkboxes = []
        options = [
            ("start_paused", "Start Paused", 0),
            ("use_pyav", "Use PyAV library", 0),
            ("rotate_video", "Rotate Video (90° CW)", 0)
        ]
        for argname, description, default_value in options:
            cb_var = tk.IntVar(value=default_value)
            cb = tk.Checkbutton(
                self.checkbox_frame,
                text=description,
                variable=cb_var
            )
            cb.pack(padx=5, pady=15, side=tk.LEFT)
            self.checkboxes.append((argname, cb_var))

        # Create the table frame
        self.table_frame = tk.Frame(self.root)
        self.table_frame.pack(fill="both", expand=True)

        # Create the table using pandastable
        self.pt = pandastable.Table(
            self.table_frame,
            dataframe=table_df
        )

        
        self.pt.bind("<Double-Button-1>", self.edit_video_file)
        self.pt.show()

        # Show the instructions table initially
        self.instructions_frame.pack(fill="x", pady=10)

        self.root.mainloop()

    def get_additional_processing_kwargs(self) -> Dict[str, bool]:
        kwargs = {arg: (variable.get() == 1) for (arg, variable) in self.checkboxes}
        return kwargs