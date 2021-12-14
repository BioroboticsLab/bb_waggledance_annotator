import cv2
import re
import math
import csv
import errno

# possible user inputs:

# double click - marks stop position (only once per video)
# right click - marks bee position
# space bar - pause video
# speed trackbar - speed up/ slow down video
# "5" - rewind 5 seconds
# "6" - skip forward 5 seconds
# "r" - restart video/ deletes already marked positions
# "q" - quit/ end video (still saves data to file)


def draw_template(img, cap, current_actuator, filepath):

    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(img, filepath, (600, do_video.height - 20), font, 1, (0, 0, 0), 1, cv2.LINE_AA)

    # draws actuators (temporarily removed since actuator positions are inaccurate)
    # for i in range(len(do_video.actuator_positions)):
    #     img = cv2.circle(img, (do_video.actuator_positions[i][0], do_video.actuator_positions[i][1]), 10, (255, 0, 0), 2)

    # if current_actuator[0]:
    #     img = cv2.circle(img, (current_actuator[0][0], current_actuator[0][1]), 10, (255, 0, 0), 5) # mark activated actuator

    calculate_time(cap)
    img = cv2.putText(img, "time in seconds: " + str(do_video.current_time), (20, do_video.height - 50), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
    return img


def draw_bee_positions(img, min_max_candidates, stop_position):

    for i in range(len(min_max_candidates)):
        img = cv2.circle(img, (min_max_candidates[i][0], min_max_candidates[i][1]), 5, (0, 255, 0), 2)

    if stop_position:
        img = cv2.circle(img, stop_position[0], 15, (0, 0, 255), 2)
    return img


def define_actuator_positions(filepath):
    #do_video.actuator_positions = [[1491, 311], [1114, 300], [719, 294], [314, 297], [296, 653], [716, 636], [1112, 632], [1506, 635]] # wrong order
    do_video.actuator_positions = [[1506, 635], [1112, 632], [716, 636], [296, 653], [314, 297], [719, 294], [1114, 300], [1491, 311]]
    # preliminary

    do_video.actuators = ["mux0", "mux1", "mux2", "mux3", "mux4", "mux5", "mux6", "mux7", "muxa"]

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
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("Frame")

    def nothing(x):  # create Trackbar doesn't work without this
        pass
    cv2.createTrackbar("Speed", "Frame", 10, 50, nothing)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.createTrackbar("Frame", "Frame", 0, total_frames, nothing)


def calculate_time(cap):
    frame_timestamp_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    do_video.current_time = round(frame_timestamp_msec / 1000, 2)


def calculate_distance(bee_coord, current_actuator):
    # euclidian distance formula: sqrt((x2 - x1)² + (y2 - y1)²)
    distance = math.sqrt((current_actuator[0][0] - bee_coord[0])**2 + (current_actuator[0][1] - bee_coord[1])**2)
    return distance


def calculate_min_max(min_max_candidates, current_actuator):
    min_max_distances = []

    max_distance = 0
    min_distance = 1000000  # arbitrary large number

    for i in range(len(min_max_candidates)):
        distance = calculate_distance(min_max_candidates[i], current_actuator)
        if max_distance <= distance:
            max_distance = distance
        if min_distance >= distance:
            min_distance = distance

    min_max_distances.append([round(min_distance, 2), round(max_distance, 2)])

    return min_max_distances


def output_data(min_max_candidates, min_max, stop_position, filepath, mux_index):

    print("marked bee positions: " + str(min_max_candidates))
    print("bee position timestamps: " + str(do_video.bee_pos_timestamps))
    print("stop coordinate: " + str(stop_position))
    print("stop time: " + str(do_video.stopping_time))    
    print("min/max distances to actuator: " + str(min_max))

    # write to csv
    header = ['video name', 'activated actuator', 'min/max distances', 'stop coordinate', 'stop time', 'marked bee positions', 'bee position timestamps']
    data = [filepath, do_video.actuators[mux_index], min_max, stop_position, do_video.stopping_time, min_max_candidates, do_video.bee_pos_timestamps]
    
    try:
        file = open('data_analysis_23092021.csv', "x", newline='')
        writer = csv.writer(file)
        writer.writerow(header)
        file.close()
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise   
        pass
    finally:
        with open("data_analysis_23092021.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)


def do_video():
    # the 10 videos
    filepath = "23092021_08_01_22_2000HZ_muxa.mp4"
    # filepath = "23092021_11_36_02_2000HZ_muxa.mp4"
    # filepath = "24092021_09_49_34_2000HZ_mux0.mp4"
    # filepath = "24092021_09_45_15_2000HZ_mux3.mp4"
    # filepath = "23092021_07_45_57_2000HZ_muxa.mp4"
    # filepath = "28092021_10_27_18_2000HZ_mux2.mp4"
    # filepath = "23092021_08_16_43_2000HZ_mux0.mp4"
    # filepath = "23092021_10_33_04_2000HZ_mux4.mp4"
    # filepath = "30092021_12_01_02_2000HZ_mux7.mp4"
    # filepath = "24092021_08_20_20_2000HZ_mux0.mp4"


    min_max_candidates = []
    stop_position = []

    do_video.current_time = 0
    do_video.stopping_time = 0
    do_video.bee_pos_timestamps = []
    
    cap = cv2.VideoCapture(filepath)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

    do_video.actuator_positions = []
    do_video.actuators = []

    current_actuator = []    
    actuator_pos, mux_index = define_actuator_positions(filepath)
    current_actuator.append(actuator_pos)
    
    setup(cap)

    while current_frame + 8 < total_frames:  # +8 for "error while decoding MB 57 57, bytestream -8"
        ret, frame = cap.read()
        if ret == True:

            speed = cv2.getTrackbarPos("Speed", "Frame")

            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

            do_video.width = int(cap.get(3))
            do_video.height = int(cap.get(4))
            
            frame = draw_template(frame, cap, current_actuator, filepath)
            frame = draw_bee_positions(frame, min_max_candidates, stop_position)
            cv2.setTrackbarPos("Frame", "Frame", int(current_frame))

            key = cv2.waitKey(speed)
            if key == ord('q'):  # press q to exit video
                break
            elif key == ord(' '):  # spacebar as pause button
                cv2.waitKey(0)
            elif key == ord('r'):  # press r to restart video (and delete bee/stop positions)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                min_max_candidates.clear()
                stop_position.clear()
                do_video.stopping_time = 0
                do_video.bee_pos_timestamps.clear()
            elif key == ord('5'):  # rewind ~5 seconds
                current_frame -= 150
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            elif key == ord('6'):  # fast forward ~5 seconds
                current_frame += 150
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

            def current_bee_position(event, x, y, flags, frame):
                nonlocal min_max_candidates
                if event == cv2.EVENT_FLAG_RBUTTON:  # right click
                    min_max_candidates.append([x, y])
                    do_video.bee_pos_timestamps.append(do_video.current_time)
                elif event == cv2.EVENT_LBUTTONDBLCLK and not stop_position:  # double click
                    stop_position.append((x, y))
                    min_max_candidates.append([x, y])  # since stop position can also be a min or max distance
                    do_video.stopping_time = do_video.current_time
                    do_video.bee_pos_timestamps.append(do_video.current_time)
                
            cv2.setMouseCallback("Frame", current_bee_position, frame)
            cv2.imshow("Frame", frame)
        else:
            break

    min_max = []
    if min_max_candidates and current_actuator[0]:
        min_max = calculate_min_max(min_max_candidates, current_actuator)
    
    cap.release()
    cv2.destroyAllWindows()

    # store dataset in file
    output_data(min_max_candidates, min_max, stop_position, filepath, mux_index)


if __name__ == "__main__":

    do_video()
    # To-do: loop through multiple files -> more efficient
