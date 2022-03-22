import streamlit as st
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

from lib.Pipeline import DataSet
import argparse	

ds = DataSet(
    directory="datasets/LookDataSet",
    extension="jpg",
    size=40,
    slope_limit=.5,
    intercept_limit=.164
)

ds.load_model(name='model_v1',train=False)

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = ds.modify_image(img)
        return img


webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)