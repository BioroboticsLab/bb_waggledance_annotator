import cv2
import dataclasses
import pandas
import re
# import math
import numpy as np
import csv
import errno

import argparse, sys, os.path

# possible user inputs:

# double click - marks stop position (only once per video)
# right click - marks bee position
# space bar - pause video
# speed trackbar - speed up/ slow down video
# "5" - rewind 5 seconds
# "6" - skip forward 5 seconds
# "r" - restart video/ deletes already marked positions
# "q" - quit/ end video (still saves data to file)

# global, for skip forward/back
FPS = 30


def get_output_filename(filepath):
    # Currently the output filename is independent of the video filepath.
    return "data_analysis_23092021.csv"

def get_csv_writer_options():
    return dict(
        quoting=csv.QUOTE_ALL,
        quotechar='"',
        delimiter=","
    )

@dataclasses.dataclass
class AnnotatedPosition:
    frame: int
    x: int
    y: int

    # Angle is annotated in vector notation in the same coordinate system as the x,y coordinates.
    # This hopefully will prevent some confusion about the origin of an angle in image coordinates.
    u: float = 0.0
    v: float = 0.0


class Annotations:

    @staticmethod
    def get_annotation_index_for_frame(list, frame):
        index = [idx for idx, position in enumerate(list) if position.frame == frame]
        assert len(index) < 2
        if len(index) == 0:
            return None
        return index[0]

    def __init__(self):
        self.raw_thorax_positions = []
        self.waggle_starts = []

    def update_thorax_position(self, frame, x, y):
        existing_index = Annotations.get_annotation_index_for_frame(self.raw_thorax_positions, frame)
        if existing_index is not None:
            self.raw_thorax_positions[existing_index].x = x
            self.raw_thorax_positions[existing_index].y = y
        else:
            self.raw_thorax_positions.append(AnnotatedPosition(frame, x, y))

    def update_waggle_start(self, frame, x, y, u=np.nan, v=np.nan):
        existing_index = Annotations.get_annotation_index_for_frame(self.waggle_starts, frame)
        if existing_index is not None:
            self.waggle_starts[existing_index].x = x
            self.waggle_starts[existing_index].y = y
            if not pandas.isnull(u):
                self.waggle_starts[existing_index].u = u
                self.waggle_starts[existing_index].v = v
        else:
            self.waggle_starts.append(AnnotatedPosition(frame, x, y, u, v))

    def update_waggle_direction(self, frame, to_x, to_y):
        existing_index = Annotations.get_annotation_index_for_frame(self.waggle_starts, frame)
        if existing_index is None:
            return
        direction = np.array([to_x - self.waggle_starts[existing_index].x,
                              to_y - self.waggle_starts[existing_index].y],
                             dtype=np.float64)
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
        return np.around(distances.min(), decimals=2), np.around(distances.max(), decimals=2)

    @staticmethod
    def load(filepath):

        def parse_string_list(values):
            if isinstance(values, list):
                values = map(ast.literal_eval, values)
                values = list(itertools.chain(*values))
            else:
                values = ast.literal_eval(values)
            return values

        # Try to read in old annotated data for this video, but don't fail fatally if anything happens.
        try:
            annotations = Annotations()

            import ast
            import pathlib
            import itertools
            # from collections import defaultdict

            annotations_df = pandas.read_csv(get_output_filename(filepath), decimal=".", **get_csv_writer_options())

            def to_leaf_name(path):
                return pathlib.Path(path).name
            annotations_df = annotations_df[annotations_df["video name"].apply(to_leaf_name) == to_leaf_name(filepath)]

            for all_xy, all_frames in annotations_df[["marked bee positions", "bee position timestamps"]].itertuples(index=False):
                all_xy = parse_string_list(all_xy)
                all_frames = parse_string_list(all_frames)

                for (xy, frame) in zip(all_xy, all_frames):
                    annotations.update_thorax_position(frame, xy[0], xy[1])

            for all_xy, all_frames, all_uv in annotations_df[["waggle start positions", "waggle start timestamps", "waggle directions"]].itertuples(index=False):
                all_xy = parse_string_list(all_xy)
                all_uv = parse_string_list(all_uv)
                all_frames = parse_string_list(all_frames)

                for (xy, frame, uv) in zip(all_xy, all_frames, all_uv):
                    annotations.update_waggle_start(frame, xy[0], xy[1], uv[0], uv[1])

            return annotations
        except Exception as e:
            print("Could not read old annotations. Continuing normally. Error: {}".format(str(e)))
            return None


def draw_template(img, cap, current_actuator, filepath):

    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(img, filepath, (600, do_video.height - 20), font, 1, (0, 0, 0), 1, cv2.LINE_AA)

    # draws actuators (temporarily removed since actuator positions are inaccurate)
    # for i in range(len(do_video.actuator_positions)):
    #     img = cv2.circle(img, (do_video.actuator_positions[i][0], do_video.actuator_positions[i][1]), 10, (255, 0, 0), 2)

    # if current_actuator[0]:
    #     img = cv2.circle(img, (current_actuator[0][0], current_actuator[0][1]), 10, (255, 0, 0), 5) # mark activated actuator

    calculate_time(cap)
    img = cv2.putText(img, "time in seconds: " + str(do_video.current_time),
                      (20, do_video.height - 50), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

    return img


def draw_bee_positions(img, annotations, current_frame, is_old_annotations=False, hide_past_annotations=False):

    colormap = dict(thorax_position=(0, 255, 0),
                    thorax_position_100_frames=(0, 0, 255),
                    waggle_start=(0, 255, 255))
    if is_old_annotations:
        colormap = dict(thorax_position=(200, 200, 200),
                        thorax_position_100_frames=(200, 200, 255),
                        waggle_start=(200, 255, 255))

    if annotations.raw_thorax_positions:
        last_marker_frame = max([p.frame for p in annotations.raw_thorax_positions])

    for position in annotations.raw_thorax_positions:
        is_last_marker = position.frame == last_marker_frame
        is_in_current_frame = current_frame == position.frame

        if (not is_last_marker and not is_in_current_frame) and hide_past_annotations:
            continue

        radius = 5 if not is_in_current_frame else 10
        img = cv2.circle(img, (position.x, position.y), radius, colormap["thorax_position"], 2)

        # We want to mark the bee 100 frames after the last thorax marking,
        # so highlight the last one at the 100 frames mark.
        if is_last_marker and current_frame > position.frame:
            size = radius
            if position.frame == current_frame - 100:
                size = radius * 6
            img = cv2.drawMarker(img, (position.x, position.y),
                                 colormap["thorax_position_100_frames"],
                                 markerType=cv2.MARKER_STAR, markerSize=size)

    if annotations.waggle_starts:
        last_waggle_start_frame = max([p.frame for p in annotations.waggle_starts])

    for position in annotations.waggle_starts:
        is_last_marker = position.frame == last_waggle_start_frame
        is_in_current_frame = current_frame == position.frame

        if (not is_last_marker and not is_in_current_frame) and hide_past_annotations:
            continue

        radius = 2 if not is_in_current_frame else 5
        length = 25 if not is_in_current_frame else 50
        img = cv2.circle(img, (position.x, position.y), radius, colormap["waggle_start"], 2)

        if not pandas.isnull(position.u) and not (np.abs(position.u) < 1e-4 and np.abs(position.v) < 1e-4):
            direction = np.array([position.u, position.v])
            direction = (direction / np.linalg.norm(direction)) * length
            direction = direction.astype(int)

            img = cv2.arrowedLine(
                img,
                (position.x - direction[0], position.y - direction[1]),
                (position.x + direction[0], position.y + direction[1]),
                colormap["waggle_start"], thickness=1)

    return img


def define_actuator_positions(filepath):
    # do_video.actuator_positions = [
    #     [1491, 311], [1114, 300], [719, 294], [314, 297],
    #     [296, 653], [716, 636], [1112, 632], [1506, 635]] # wrong order
    do_video.actuator_positions = [
        [1506, 635], [1112, 632], [716, 636], [296, 653],
        [314, 297], [719, 294], [1114, 300], [1491, 311]
    ]
    # preliminary

    do_video.actuators = ["mux0", "mux1", "mux2", "mux3", "mux4",
                          "mux5", "mux6", "mux7", "muxa"]

    for i in range(len(do_video.actuators)):
        if re.search(do_video.actuators[i], filepath):
            if do_video.actuators[i] != "muxa":
                current_actuator = do_video.actuator_positions[i]
                index = i
            else:
                current_actuator = []
                index = 8

    return current_actuator, index


def setup(cap):
    cv2.namedWindow("Frame", cv2.WINDOW_GUI_NORMAL)
    # cv2.namedWindow("Frame")

    def nothing(x):  # create Trackbar doesn't work without this
        pass
    cv2.createTrackbar("Speed", "Frame", 30, 60, nothing)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.createTrackbar("Frame", "Frame", 0, total_frames, nothing)


def calculate_time(cap):
    frame_timestamp_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    do_video.current_time = round(frame_timestamp_msec / 1000, 2)


def output_data(annotations: Annotations, min_max, filepath, mux_index):

    print("marked bee positions: " + str([(p.frame, p.x, p.y) for p in annotations.raw_thorax_positions]))
    print("min/max distances to actuator: " + str(min_max))

    # Write to csv
    header = ['video name', 'activated actuator', 'min/max distances',
              'marked bee positions', 'bee position timestamps',
              'waggle start positions', 'waggle start timestamps', 'waggle directions'
              ]
    thorax_xy = [(p.x, p.y) for p in annotations.raw_thorax_positions]
    thorax_frames = [p.frame for p in annotations.raw_thorax_positions]
    waggle_xy = [(p.x, p.y) for p in annotations.waggle_starts]
    waggle_frames = [p.frame for p in annotations.waggle_starts]
    waggle_directions = [(p.u, p.v) for p in annotations.waggle_starts]
    data = [filepath, do_video.actuators[mux_index], min_max,
            thorax_xy, thorax_frames,
            waggle_xy, waggle_frames, waggle_directions
            ]

    output_filepath = get_output_filename(filepath)
    is_first_entry = not os.path.exists(output_filepath)
    
    with open(output_filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, **get_csv_writer_options())

        if is_first_entry:
            writer.writerow(header)
        writer.writerow(data)


def calc_frames_to_move(k32: int, debug: bool = False) -> int:
    ''' from extended keycode, select nframes to advance or rewind.'''
    # hmm modifiers are actually 17, 19, 20 bits away
    ctrl  = (k32 & (1 << 18)) >> 18
    shift = (k32 & (1 << 16)) >> 16
    alt   = (k32 & (1 << 19)) >> 19
    modifiers = f"{ctrl:0x} {shift:0x} {alt:0x}"
    if k32 != -1 and debug:
        key = k32 & 0xFF
        print('\033[34m' + f"KEY: {k32} 0x{k32:2x} |{key} 0x{key:2x} |{modifiers}|" + '\033[0m')
    nframes = 1
    if shift: nframes = int(FPS * 1)
    elif ctrl: nframes = int(FPS * 5)
    elif alt: nframes = int(5)

    return nframes


def do_video(filepath: str, debug: bool = False):
    # the 10 videos
    # filepath = "23092021_08_01_22_2000HZ_muxa.mp4"
    # filepath = "23092021_11_36_02_2000HZ_muxa.mp4"
    # filepath = "24092021_09_49_34_2000HZ_mux0.mp4"
    # filepath = "24092021_09_45_15_2000HZ_mux3.mp4"
    # filepath = "23092021_07_45_57_2000HZ_muxa.mp4"
    # filepath = "28092021_10_27_18_2000HZ_mux2.mp4"
    # filepath = "23092021_08_16_43_2000HZ_mux0.mp4"
    # filepath = "23092021_10_33_04_2000HZ_mux4.mp4"
    # filepath = "30092021_12_01_02_2000HZ_mux7.mp4"
    # filepath = "24092021_08_20_20_2000HZ_mux0.mp4"

    annotations = Annotations()
    old_annotations = Annotations.load(filepath)

    do_video.current_time = 0  # Time for display purposes only.

    cap = cv2.VideoCapture(filepath)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    original_frame_image = None
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    assert current_frame == 0
    is_in_pause_mode = False
    is_in_draw_vector_mode = False
    hide_past_annotations = False
    normalize_contrast = False

    do_video.actuator_positions = []
    do_video.actuators = []

    current_actuator = []
    actuator_pos, mux_index = define_actuator_positions(filepath)
    current_actuator.append(actuator_pos)

    setup(cap)

    def move_frame_count(offset):
        nonlocal current_frame
        nonlocal cap
        nonlocal total_frames
        nonlocal original_frame_image

        current_frame = min(max(current_frame + offset, 0), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        # Clear current raw frame so we fetch a new one even if in pause mode.
        original_frame_image = None

    # Set up mouse interaction callback.
    def on_mouse_event(event, x, y, flags, user_data):
        nonlocal current_frame, is_in_draw_vector_mode, is_in_pause_mode

        if event == cv2.EVENT_FLAG_RBUTTON:  # right click
            annotations.update_thorax_position(current_frame, x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:  # double click
            annotations.update_waggle_start(current_frame, x, y)
            is_in_draw_vector_mode = True
            # Force pause for drawing. Otherwise, the updates might affect the wrong arrow.
            if not is_in_pause_mode:
                # Additional marker to True/False for "forced pause".
                is_in_pause_mode = 2
        elif event == cv2.EVENT_MOUSEMOVE:
            if is_in_draw_vector_mode:
                annotations.update_waggle_direction(current_frame, x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            is_in_draw_vector_mode = False
            if is_in_pause_mode == 2:
                is_in_pause_mode = False
            annotations.update_waggle_direction(current_frame, x, y)

    # Manually create the window with a flag that should disable right-click context menu.
    cv2.setMouseCallback("Frame", on_mouse_event)

    while current_frame + 8 < total_frames:  # +8 for "error while decoding MB 57 57, bytestream -8"
        try:
            speed_fps = cv2.getTrackbarPos("Speed", "Frame")
        except:
            # Might fail when window is already deconstructed.
            # But then, the application is being terminated anyway.
            break
        has_valid_frame = True

        # Fetch a new frame if either the speed is > 0 or we are not currently pausing the video playback.
        if (speed_fps > 0 and not is_in_pause_mode) or original_frame_image is None:
            # Make sure to read the current frame number only BEFORE fetching an image.
            # cap.read() will advance the frame number after reading.
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            has_valid_frame, original_frame_image = cap.read()
        frame = original_frame_image.copy()

        if not has_valid_frame:
            break

        do_video.width = int(cap.get(3))
        do_video.height = int(cap.get(4))

        if normalize_contrast:
            # Use in-place operations for a slightly better performance.
            np.subtract(frame, frame.min(), out=frame)
            frame = frame.astype(np.float32)
            np.divide(frame, frame.max() / 255.0, out=frame)
            frame = frame.astype(np.uint8)

        frame = draw_template(frame, cap, current_actuator, filepath)
        frame = draw_bee_positions(frame, annotations, current_frame=current_frame, hide_past_annotations=hide_past_annotations)
        if old_annotations and not hide_past_annotations:
            frame = draw_bee_positions(frame, old_annotations, current_frame=current_frame, is_old_annotations=True)

        cv2.setTrackbarPos("Frame", "Frame", int(current_frame))

        if speed_fps <= 0 or is_in_pause_mode:
            # Arbitrary delay > 0, because we need to update the UI even in pause mode.
            delay_ms = 50
        else:
            delay_ms = max(1, int(1000 / speed_fps))

        # key = cv2.waitKey(delay_ms)
        k32 = cv2.waitKeyEx(delay_ms)
        key = k32 & 0xFF

        nframes = calc_frames_to_move(k32, debug)

        if key == ord('q'):  # press q to exit video
            break
        elif key == ord(' '):  # spacebar as pause button
            is_in_pause_mode = not is_in_pause_mode
        elif key == ord('r'):  # press r to restart video (and delete bee/stop positions)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            annotations.clear()
        elif key in [ord('a'), ord('A'), 0x51]:  # 0x51 is left arrow
            move_frame_count(-nframes)
        elif key in [ord('d'), ord('D'), 0x53]:  # 0x53 is right arrow (weird, should be 37-40 LURD)
            move_frame_count(+nframes)
        # elif key == ord('a'):
        #     move_frame_count(-1)
        # elif key == ord('d'):
        #     move_frame_count(+1)
        elif key == ord('w'):
            move_frame_count(+25)
        elif key == ord('s'):
            move_frame_count(-25)
        elif key == ord('5'):  # rewind ~5 seconds
            move_frame_count(-150)
        elif key == ord('6'):  # fast forward ~5 seconds
            move_frame_count(+150)
        elif key == ord("+"):
            cv2.setTrackbarPos("Speed", "Frame", min((speed_fps // 10) * 10 + 10, 60))
        elif key == ord("-"):
            cv2.setTrackbarPos("Speed", "Frame", max((speed_fps // 10) * 10 - 10, 0))
        elif key == ord("h"):
            hide_past_annotations = not hide_past_annotations
        elif key == ord("c"):
            normalize_contrast = not normalize_contrast

        cv2.imshow("Frame", frame)

    min_max = annotations.calculate_min_max_thorax_distance_to_actuator(current_actuator[0]) if len(current_actuator) > 0 else None

    cap.release()
    cv2.destroyAllWindows()

    # store dataset in file
    output_data(annotations, min_max, filepath, mux_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-v', '--verb', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('-p', '--path', type=str, default="./",
                        help="select path to video files")
    parser.add_argument('-n', '--num', type=int, default=None,
                        help='select the video number to analyse')
    args = parser.parse_args()

    # the 10 videos
    files = [
        '23092021_07_45_57_2000HZ_muxa.mp4',
        '23092021_08_01_22_2000HZ_muxa.mp4',
        '23092021_08_16_43_2000HZ_mux0.mp4',
        '23092021_10_33_04_2000HZ_mux4.mp4',
        '23092021_11_36_02_2000HZ_muxa.mp4',
        '24092021_08_20_20_2000HZ_mux0.mp4',
        '24092021_09_45_15_2000HZ_mux3.mp4',
        '24092021_09_49_34_2000HZ_mux0.mp4',
        '28092021_10_27_18_2000HZ_mux2.mp4',
        '30092021_12_01_02_2000HZ_mux7.mp4']
    if args.num is None or args.num < 0 or args.num > len(files):
        print(f"[E] we have {len(files)} files, pick one with -n <N>:")
        print("\n".join([f"  {i:3d}:  {f}" for i, f in enumerate(files)]))
        sys.exit(1)  # lazy exit

    vidfile = files[args.num]
    print(f"[I] index {args.num} -> file {vidfile}")
    filepath = os.path.join(args.path, vidfile)
    if not os.path.exists(filepath):
        raise RuntimeError("[E] file {filepath} not available. check --path option")

    do_video(filepath, args.debug)
    # To-do: loop through multiple files -> more efficient
