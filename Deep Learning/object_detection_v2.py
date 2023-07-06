import datetime
from ultralytics import YOLO
import cv2
from helper import create_video_writer

# streamlit imports
import streamlit as st
from PIL import Image
import numpy as np
import cv2

# time import
import time

# text to speech imports
from gtts import gTTS
from io import BytesIO

# youtube imports
from pytube import YouTube
import youtube_dl
import tempfile


# define some constants
CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)


def format_class_name(class_name, count):
    class_name = class_name.capitalize()
    return class_name + "s" if count > 1 else class_name


def generate_description(data):
    # Base string
    base_string = "In the picture there "

    if len(data) == 0:
        return "No objects detected"

    # Constructing the list of items
    items = [
        f"{count} {format_class_name(class_name, count)}"
        for class_name, count in data.items()
    ]

    # Combining the items into a single string with commas and 'and' before the last item
    if len(items) > 1:
        last_item = items.pop()
        formatted_items = ", ".join(items) + " and " + last_item
    else:
        formatted_items = items[0]

    # Combining base_string with formatted_items
    final_string = base_string + "is " + formatted_items + "."

    return final_string


st.title("DeText2Speech")

frame_placeholder = st.empty()

# Check if 'is_playing' key is in session_state
if "is_playing" not in st.session_state:
    st.session_state["is_playing"] = False

# Check if 'button_pressed' key is in session_state
if "button_pressed" not in st.session_state:
    st.session_state["button_pressed"] = False

# Create Play/Pause button
if st.button("Play/Pause"):
    # Toggle is_playing state
    st.session_state["is_playing"] = not st.session_state["is_playing"]
    # Indicate that the button has been pressed
    st.session_state["button_pressed"] = True

# initialize the video capture object
video_cap = cv2.VideoCapture(0)
# initialize the video writer object
writer = create_video_writer(video_cap, "output.mp4")

# load the pre-trained YOLOv8n model
model = YOLO("yolov8n.pt")

while True:
    # start time to compute the fps
    start = datetime.datetime.now()

    ret, frame = video_cap.read()

    # if there are no more frames to process, break out of the loop
    if not ret:
        break

    # run the YOLO model on the frame
    detections = model(frame)[0]

    # loop over the detections
    for data in detections.boxes.data.tolist():
        # extract the confidence (i.e., probability) associated with the detection
        confidence = data[4]

        # filter out weak detections by ensuring the
        # confidence is greater than the minimum confidence
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        # if the confidence is greater than the minimum confidence,
        # draw the bounding box on the frame
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])

        # use different colours for different classes
        if detections.names[0] == "person":
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        elif detections.names[0] == "car":
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    # end time to compute the fps
    end = datetime.datetime.now()
    # show the time it took to process 1 frame
    total = (end - start).total_seconds()
    print(f"Time to process 1 frame: {total * 1000:.0f} milliseconds")

    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / total:.2f}"
    cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    if len(detections.names) > 0:
        # Loop through all detected objects
        for i, box in enumerate(detections.boxes):
            # Extract the bounding box coordinates from the 'xyxy' attribute of 'box'
            xmin, ymin, xmax, ymax = box.xyxy[0].numpy()

            # Get class index from detections and map it to actual name
            class_index = int(box.cls[0].item())
            class_name = detections.names.get(class_index, "Unknown")

            # Convert class_name to string if it's not
            class_name = str(class_name)

            # Put the name of the detected object on the bounding box
            cv2.putText(
                frame,
                class_name,
                (int(xmin), int(ymin) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                8,
            )

    # show the frame to streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, channels="RGB")
    if st.session_state["button_pressed"]:
        # Reset the button_pressed state
        st.session_state["button_pressed"] = False

        object_dict = {}

        for i, box in enumerate(detections.boxes):
            class_index = int(box.cls[0].item())
            class_name = detections.names.get(class_index, "Unknown")
            # add class name to dictionary. If it already exists, increment the count
            if class_name in object_dict:
                object_dict[class_name] += 1
            else:
                object_dict[class_name] = 1

        description = generate_description(object_dict)

        sound_file = BytesIO()
        tts = gTTS(description, lang="en")
        tts.write_to_fp(sound_file)

        st.audio(sound_file)

        time.sleep(10)

    writer.write(frame)
    if cv2.waitKey(1) == ord("q"):
        break

video_cap.release()
writer.release()
cv2.destroyAllWindows()
