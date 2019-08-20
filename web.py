# -- (c) 2019 Takahito Murakami, Student council of kasuga UTsukuba - #


import cv2
import numpy as np

#ArUcoマーカーセットアップ
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

# カメラキャリブレーションパラメータ（MacBook air 2018)
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))  #歪み

#カメラキャプチャ
cap = cv2.VideoCapture(0)


def draw_axes(img, rvec, tvec, size=0.05): #3D座標軸を描画 
    axis = np.float32([[size, 0, 0], [0, size, 0],
                      [0, 0, size]]).reshape(-1, 3)
    imgpts, jac = cv2.projectPoints(
        axis, rvec, tvec, camera_matrix, dist_coeffs)

    #中点を2Dへ変換
    origin, _ = cv2.projectPoints(
        np.array([[0.0, 0.0, 0.0]]), rvec, tvec, camera_matrix, dist_coeffs)
    origin = tuple(origin[0].ravel().astype(int))
    imgpts = imgpts.reshape(-1, 2).astype(int)

    #軸を描画する
    img = cv2.line(img, origin, tuple(imgpts[0]), (0, 0, 255), 5)  # X軸
    img = cv2.line(img, origin, tuple(imgpts[1]), (0, 255, 0), 5)  # Y軸
    img = cv2.line(img, origin, tuple(imgpts[2]), (255, 0, 0), 5)  # Z軸
    return img


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)

    if ids is not None and len(ids) >= 2: #2つのマーカーが認識された時の姿勢を推定
        rvec1, tvec1, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners[0], 0.05, camera_matrix, dist_coeffs)
        rvec2, tvec2, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners[1], 0.05, camera_matrix, dist_coeffs)

        #2つのマーカーの中点を計算
        midpoint = (tvec1[0] + tvec2[0]) / 2.0

        #中点に座標軸を描画
        frame = draw_axes(frame, rvec1[0], midpoint)

        # マーカーの中心にも座標軸を描画
        frame = draw_axes(frame, rvec1[0], tvec1[0])
        frame = draw_axes(frame, rvec2[0], tvec2[0])

        #マーカーをフレームに描画
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    #フレームを表示
    cv2.imshow('AR Marker Tracking with 3D Axes at Midpoint', frame)

    #'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
