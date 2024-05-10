import Utils
import Segmentation
import GaFit

import cv2
import os


img = cv2.imread(os.path.join(os.getcwd(), 'img.jpg'))
landmarks = Utils.get_landmarks(img)
left_bbox, right_bbox = Utils.get_info(img, landmarks)
eyes = Utils.crop_eyes(img, left_bbox, right_bbox)

combinedArr, contourPointsArr, maskScleraArr, maskIrisArr = Segmentation.get_eyes_seg_result(eyes)
solution, _, __ = GaFit.run(contourPointsArr)

# 注意: solution為 List 長度5 當中包含座標僅代表在eyes[0] eyes[1]中 eyes[0].shape = (512, 512)
# soluion[0] 左眼虹膜中心點x
# soluion[1] 左眼虹膜中心點y
# soluion[2] 右眼虹膜中心點x
# soluion[3] 右眼虹膜中心點y
# soluion[4] 雙眼虹膜半徑(pixel)

# 換算原圖大小
scale = 512/eyes[0].shape[0]
x, y = int(solution[0] / scale), int(solution[1] / scale)
x1, y1 = int(solution[2] / scale), int(solution[3] / scale)
radius = int(solution[4] / scale)

cv2.circle(img, (x + left_bbox[0], y + left_bbox[1]), 1, (0, 0, 255), 2)
cv2.circle(img, (x + left_bbox[0], y + left_bbox[1]), radius, (0, 0, 255), 2)
cv2.circle(img, (x1 + right_bbox[0], y1 + right_bbox[1]), 1, (0, 0, 255), 2)
cv2.circle(img, (x1 + right_bbox[0], y1 + right_bbox[1]), radius, (0, 0, 255), 2)

cv2.imwrite(os.path.join(os.getcwd(), 'res.jpg'), img)
