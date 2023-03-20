import cv2
import numpy as np

# set red thresh
lower_red = np.array([156, 43, 46])
upper_red = np.array([180, 255, 255])
# set yellow thresh
lower_yellow = np.array([26, 43, 46])
upper_yellow = np.array([34, 255, 255])
# set purple thresh
lower_purple = np.array([125, 43, 46])
upper_purple = np.array([155, 255, 255])

# 视频录制
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("D:/Documents/Python/video/synthesis_10fps6.avi", fourcc, 10, (1280, 720), True)

def calculate_pixels(image):
    hsv_model = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # get red mask
    mask = cv2.inRange(hsv_model, lower_red, upper_red)
    # 返回像素值不为零的数(即白色像素点个数)
    red_pixels = cv2.countNonZero(mask)
    # 计算相应颜色像素在图片中的比例
    red_rate = red_pixels / (image.shape[0] * image.shape[1])
    print("red_rate:%f" % red_rate)

    # get yellow mask
    mask = cv2.inRange(hsv_model, lower_yellow, upper_yellow)
    # 返回像素值不为零的数(即白色像素点个数)
    yellow_pixels = cv2.countNonZero(mask)
    # 计算相应颜色像素在图片中的比例
    yellow_rate = yellow_pixels / (image.shape[0] * image.shape[1])
    print("yellow_rate:%f" % yellow_rate)

    # get purple mask
    mask = cv2.inRange(hsv_model, lower_purple, upper_purple)
    # 返回像素值不为零的数(即白色像素点个数)
    purple_pixels = cv2.countNonZero(mask)
    # 计算相应颜色像素在图片中的比例
    purple_rate = purple_pixels / (image.shape[0] * image.shape[1])
    print("purple_rate:%f" % purple_rate)

    color_dict = {"red": red_rate, "yellow": yellow_rate, "purple": purple_rate}
    # print(max(color_dict, key=color_dict.get))
    color = max(color_dict, key=color_dict.get)

    #cv2.putText(frame, color, np.int32(points)[0][0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    print(color)
    #cv2.putText(frame, color, (int(center_x - 1.5 * width / 2), int(center_y - 2 * height / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 2)
    if color == "red":
        cv2.rectangle(frame, (int(center_x - 1.5 * width / 2), int(center_y - 2 * height / 2)),
                      (int(center_x + 1.5 * width / 2), int(center_y + 2 * height / 2)), (0, 0, 255), 2)
        cv2.putText(frame, color, (int(center_x - 1.5 * width / 2), int(center_y - 2 * height / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    elif color == "yellow":
        cv2.rectangle(frame, (int(center_x - 1.5 * width / 2), int(center_y - 2 * height / 2)),
                      (int(center_x + 1.5 * width / 2), int(center_y + 2 * height / 2)), (0, 255, 255), 2)
        cv2.putText(frame, color, (int(center_x - 1.5 * width / 2), int(center_y - 2 * height / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)

    elif color == "purple":
        cv2.rectangle(frame, (int(center_x - 1.5 * width / 2), int(center_y - 2 * height / 2)),
                      (int(center_x + 1.5 * width / 2), int(center_y + 2 * height / 2)), (255, 0, 255), 2)
        cv2.putText(frame, color, (int(center_x - 1.5 * width / 2), int(center_y - 2 * height / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 2)


if __name__ == "__main__":
    capture = cv2.VideoCapture(1)
    capture.set(3, 1280)
    capture.set(4, 720)

    # 初始化QR检测器
    QRcode = cv2.QRCodeDetector()
    pre_codeinfo = ["", "", "", "", ""]

    while True:
        ret, frame = capture.read()
        codeinfo, decoded_info, points, straight_qrcode = QRcode.detectAndDecodeMulti(frame)

        # 如果检测到二维码且二维码内容不为空
        if (points is not None) and (decoded_info != ""):
            for i in range(len(points)):
                x_min = np.int32(points)[i][0][0]
                x_max = np.int32(points)[i][1][0]
                y_min = np.int32(points)[i][1][1]
                y_max = np.int32(points)[i][2][1]

                # 二维码中心点x坐标
                center_x = (x_min + x_max) / 2
                # 二维码中心点y坐标
                center_y = (y_min + y_max) / 2
                # 二维码宽度(宽度和高度两者往往相等)
                width = x_max - x_min
                # 二维码高度
                height = y_max - y_min

                cv2.drawContours(frame, [np.int32(points)[i]], 0, (0, 0, 255), 2)
                # print(decoded_info[i])
                # 判断检测到的是否为同一个二维码
                if decoded_info[i] != pre_codeinfo[i]:
                    print(decoded_info[i])
                    pre_codeinfo[i] = decoded_info[i]
                    cv2.putText(frame, decoded_info[i], np.int32(points)[i][0], cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                (0, 255, 0), 2)

                else:
                    cv2.putText(frame, pre_codeinfo[i], np.int32(points)[i][0], cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                (0, 255, 0), 2)

                # 截取特定区域(img[y,x])
                src = frame[int(center_y - 2 * height / 2):int(center_y + 2 * height / 2),
                      int(center_x - 1.5 * width / 2):int(center_x + 1.5 * width / 2)]
                calculate_pixels(src)

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
        # 按esc退出视频
        if c == 27:
            break