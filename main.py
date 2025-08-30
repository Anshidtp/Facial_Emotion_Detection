from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image


model = YOLO("Model/best.pt")
results = model.predict(source="0", show=True)