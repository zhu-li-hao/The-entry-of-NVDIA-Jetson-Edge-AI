import cv2
import numpy as np

capture = cv2.VideoCapture(1)
capture.set(3, 1280)
capture.set(4, 720)

# 初始化QR检测器
QRcode = cv2.QRCodeDetector()
pre_codeinfo = ["", "", "", "", ""]

# 视频录制
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("D:/Documents/Python/video/QRcode_multi_video2_20fps3.avi", fourcc, 20, (1280, 720), True)

while True:
    ret, frame = capture.read()
    codeinfo, decoded_info, points, straight_qrcode = QRcode.detectAndDecodeMulti(frame)

    if (points is not None) and (decoded_info != ""):
        print(np.int32(points))
        for i in range(len(points)):
            cv2.drawContours(frame, [np.int32(points)[i]], 0, (0, 0, 255), 2)
            print(decoded_info[i])
            if decoded_info[i] != pre_codeinfo[i]:
                print(decoded_info[i])
                pre_codeinfo[i] = decoded_info[i]
                cv2.putText(frame, decoded_info[i], np.int32(points)[i][0], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            else:
                cv2.putText(frame, pre_codeinfo[i], np.int32(points)[i][0], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        cv2.namedWindow("result", 0)
        cv2.resizeWindow("result", 640, 360)
        cv2.imshow("result", frame)
        out.write(frame)

    else:
        cv2.putText(frame, 'QRcode is not detected', (1, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.namedWindow("result", 0)
        cv2.resizeWindow("result", 640, 360)
        cv2.imshow("result", frame)
        out.write(frame)

    c = cv2.waitKey(1)
    if c == 27:
        break