import argparse
import tkinter as tk
from .ui import FileSelectorUI
from .video_capture import do_video

import cProfile
import pstats
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "-p", "--path", type=str, help="select path to video files", default='./'
    )
    args = parser.parse_args()

    folder_path = args.path

    # Create the root Tk() instance
    root = tk.Tk()
    root.withdraw()  # Hide it initially

    def on_close():
        """Handles window close event and exits gracefully."""
        print("Window closed, exiting program.")
        root.quit()  # Quit the Tkinter mainloop
        root.destroy()  # Destroy the root window to release resources

    # Bind the window close event (clicking 'X') to our on_close function
    root.protocol("WM_DELETE_WINDOW", on_close)

    while True:
        ui = FileSelectorUI(
            root,
            folder_path,
            on_filepath_selected=lambda path, **kwargs: do_video(
                root, path, args.debug, **kwargs
            ),
        )
        ui.show()

        # After ui.show(), check if the user closed the UI without selecting a video
        if ui.selected_video is None:
            print("No video selected. Exiting.")
            break
        else:
            # Continue the loop to re-open the UI
            continue

if __name__ == "__main__":
    # Initialize the cProfile profiler
    profile = cProfile.Profile()
    profile.enable()

    try:
        main()  # Run the main program
    finally:
        # Ensure profiling stats are saved or printed even if the program exits
        profile.disable()

        # Print profiling stats after program ends
        p = pstats.Stats(profile)
        p.sort_stats('cumtime').print_stats(50)  # Print top 50 functions by cumulative time