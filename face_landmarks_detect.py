import cv2
import mediapipe as mp
import yt_dlp
from fer import FER

# Initialize MediaPipe Face Mesh and Drawing utilities
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Drawing specifications
landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0),thickness=1, circle_radius=1)
text_color = (0, 255, 0)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_thickness = 2
line_type = cv2.LINE_AA
info_text = "Press 'Esc' to exit"


url = "https://youtu.be/5w3cYtJekpw?si=G-MHZrn5ahba7SXK"


# Get direct stream URL using yt-dlp
ydl_opts = {'format': 'best'}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(url, download=False)
    stream_url = info_dict['url']

# Start webcam capture
cam_video = cv2.VideoCapture(stream_url)

# Emotion detection model
detector = FER(mtcnn=False)


# Get text size for the on-screen instructions
text_size, _ = cv2.getTextSize(info_text, font, font_scale, font_thickness)

# Set up the Face Mesh model
with mp_face.FaceMesh(
    refine_landmarks=True,  
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    
    while cam_video.isOpened():
        ret, frame = cam_video.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # Convert frame to RGB (required by MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        result = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        frame_height, frame_width = frame.shape[:2]

        

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:

                # Draw facial mesh tesselation (full grid)
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=landmark_drawing_spec,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                )

                # Draw face contour lines (outline of facial features)
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

                # Draw iris landmarks (only valid if refine_landmarks=True)
                if len(face_landmarks.landmark) >= 478:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                    )
            
        
        # Emotion Detection
        emotions_result = detector.detect_emotions(frame)
        if emotions_result:
            (x, y, w, h) = emotions_result[0]["box"]
            emotions = emotions_result[0]["emotions"]

            # Get top emotion
            top_emotion = max(emotions, key=emotions.get)
            score = emotions[top_emotion]

            # Draw bounding box + emotion text
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{top_emotion} ({score:.2f})", (x, y - 10),
                        font, 0.9, (0, 255, 0), 2)

                

        # # Display info text on screen
        cv2.rectangle(frame, (5, 5), (10 + text_size[0], 10 + text_size[1]), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (10, 10 + text_size[1]), font, font_scale, text_color, font_thickness, line_type)

        # Show the frame
        cv2.imshow("Facial Landmarks Detection", frame)

        # Exit on pressing ESC
        if cv2.waitKey(10) & 0xFF == 27:
            break

# Release camera and close windows
cam_video.release()
cv2.destroyAllWindows()
