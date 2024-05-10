import mediapipe as mp
import cv2
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# 取得人臉標記點
def get_landmarks(img):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    results = face_mesh.process(image_rgb)
    landmarks = None

    if(results.multi_face_landmarks):
        landmarks = results.multi_face_landmarks[0].landmark

    return landmarks

# 取得左右眼定界框
def get_info(img, landmarks):
    if(landmarks is not None):
        temp = []
        # 右眼(bbox)
        x0 = landmarks[33].x * img.shape[1]
        y0 = landmarks[33].y * img.shape[0]
        x1 = landmarks[133].x * img.shape[1]
        y1 = landmarks[33].y * img.shape[0]
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        width0 = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2) * 1.2
        temp.append([mid_x, mid_y])

        # 左眼(bbox)
        x0 = landmarks[362].x * img.shape[1]
        y0 = landmarks[263].y * img.shape[0]
        x1 = landmarks[263].x * img.shape[1]
        y1 = landmarks[263].y * img.shape[0]
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        width1 = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2) * 1.2
        temp.append([mid_x, mid_y])


        width = max([width0, width1])
        left_bbox = (
            int(temp[1][0] - width / 2),
            int(temp[1][1] - width / 2),
            int(temp[1][0] + width / 2),
            int(temp[1][1] + width / 2)
        )

        right_bbox = (
            int(temp[0][0] - width / 2),
            int(temp[0][1] - width / 2),
            int(temp[0][0] + width / 2),
            int(temp[0][1] + width / 2)
        )
    else:
        print("error")

    return left_bbox, right_bbox

def crop_eyes(img, left_bbox, right_bbox):
    left_right_bbox_size = (
        abs(left_bbox[0] - left_bbox[2]),
        abs(left_bbox[1] - left_bbox[3]),
        abs(right_bbox[0] - right_bbox[2]),
        abs(right_bbox[1] - right_bbox[3])
    )

    size = max(left_right_bbox_size)
    left_start, left_end = (left_bbox[0], left_bbox[1]), (left_bbox[0]+size, left_bbox[1]+size)
    right_start, right_end = (right_bbox[0], right_bbox[1]), (right_bbox[0]+size, right_bbox[1]+size)

    left = img[left_start[1]: left_end[1], left_start[0]: left_end[0]]          # 挑出左眼
    right = img[right_start[1]: right_end[1], right_start[0]: right_end[0]]     # 挑出右眼

    return [left, right]

