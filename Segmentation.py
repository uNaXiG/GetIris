import onnxruntime as ort
import numpy as np
import cv2
# providers=['CUDAExecutionProvider', "CPUExecutionProvider"]
providers=["CPUExecutionProvider"]

def inference_model(model: ort.InferenceSession, img):
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    result = model.run([output_name], {input_name: img})[0]
    return result


def iris_segmenation(img):
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    input_img = (input_img - mean) / std
    input_img = input_img.astype(np.float32)
    # HWC to NCHW
    input_img = np.transpose(input_img, [2, 0, 1])
    input_img = np.expand_dims(input_img, 0)
    result = inference_model(irisSegmenationModel, input_img)
    result = result.reshape(512, 512)
    result = result.astype(np.uint8)
    return result

print("Loading Iris Segmenation Model")
irisSegmenationModel = ort.InferenceSession(
    r"segmentation\IrisSegmentation.onnx", providers=providers
)

# print("Testing Model")
# inference_model(irisSegmenationModel, np.zeros((1, 3, 512, 512), dtype=np.float32))

print(f"Models run on {ort.get_device()}")
print()

def get_eyes_seg_result(eyes):
    combinedArr = []
    contourPointsArr = []
    maskScleraArr = []
    maskIrisArr = []
    for idx, eye in enumerate(eyes):
        eye_img = eye.copy()
        eye_img = cv2.resize(eye_img, (512, 512), interpolation=cv2.INTER_AREA)

        result = iris_segmenation(eye_img)
        combined, contour, contourPoints, mask_sclera, mask_iris = findContour(result)

        combinedArr.append(combined)
        contourPointsArr.append(contourPoints)
        maskScleraArr.append(mask_sclera)
        maskIrisArr.append(mask_iris)


    return combinedArr, contourPointsArr, maskScleraArr, maskIrisArr
