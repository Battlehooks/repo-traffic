import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2 as cv
import time
from ultralytics import YOLO
fps = 20
spf = 1/fps
dy = 15
x, y = 10, 20
annot = (x, y)
model = YOLO('runs/detect/train/weights/best.pt')

st.title('Bandung SlowMo Traffic Detector')
video = 0

with st.sidebar:
    video = st.button('Video Traffic Check', use_container_width=True)
    st.button('Traffic Realtime Monitor', use_container_width=True)

btn_change = '''
    <script>
        var elements = window.parent.document.querySelectorAll(')
'''


def weighting(count) -> int:
    count = count.to('cpu')
    car = np.sum(np.where(count == 3, 2, 0))
    bus = np.sum(np.where(count == 2, 3, 0))
    truck = np.sum(np.where(count == 6, 3, 0))
    ambulance = np.sum(np.where(count == 0, 30, 0))
    motor = np.sum(np.where(count == 5, 1, 0))
    motorW = motor ** 1.25

    dicts = {
        'Density': np.sum([car, bus, truck, ambulance, motorW]),
        'Jumlah Kendaraan': np.sum([car/2, bus/3, truck/3, ambulance/30, motor]),
        'Motor': motor,
        'Mobil': car/2,
        'Bus': bus/2,
        'Truck': truck/3,
        'Ambulance': ambulance/3
    }

    return dicts


with st.container():
    url = st.text_input('Input Url', key='stream_url')
    cap = cv.VideoCapture(url)
    if url:
        font = cv.FONT_HERSHEY_SIMPLEX
        color_black = (0, 0, 0)
        color_white = (255, 255, 255)
        thick_inline = 1
        thick_outline = 2
        fontScale = .5
        stopped_btn = st.checkbox('Stop')
        plot_box = st.checkbox('Disable Box Annotate')
        annotation = st.checkbox('Disable Annotate')
        plot_box = not plot_box
        annotation = not annotation
        img = st.image([], use_column_width=True)
        while not stopped_btn:
            x, y = 10, 20
            annot = (x, y)
            current = time.time()
            ret, frame = cap.read()
            result = model(frame, imgsz=360)[0]
            frame_res = result.plot(
                labels=plot_box, boxes=plot_box, masks=plot_box, probs=plot_box)
            frame_res = cv.cvtColor(frame_res, cv.COLOR_BGR2RGB)
            weights = weighting(result.boxes.cls)
            if annotation:
                for k, v in weights.items():
                    frame_res = cv.putText(
                        frame_res, f'{k} : {v:.1f}', (
                            x, y), font, fontScale, color_black, thick_outline, cv.LINE_AA, False
                    )
                    frame_res = cv.putText(
                        frame_res, f'{k} : {v:.1f}', (
                            x, y), font, fontScale, color_white, thick_inline, cv.LINE_AA, False
                    )
                    y = y + dy
            img.image(frame_res)
            duration = time.time() - current
            if duration < spf:
                time.sleep(spf-duration)
        cap.release()
