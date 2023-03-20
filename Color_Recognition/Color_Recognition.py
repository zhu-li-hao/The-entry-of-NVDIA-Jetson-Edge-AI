import cv2
import numpy as np

# 视频录制
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("D:/Documents/Python/video/color_detect_20fps.avi", fourcc, 20, (1280, 720), True)

# set red thresh
lower_red = np.array([156, 43, 46])
upper_red = np.array([180, 255, 255])
# set yellow thresh
lower_yellow = np.array([11, 43, 46])
upper_yellow = np.array([50, 255, 255])
# set purple thresh
lower_purple = np.array([125, 43, 46])
upper_purple = np.array([155, 255, 255])

def color_detect(image):
    hsv_model = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # get red mask
    red_mask = cv2.inRange(hsv_model, lower_red, upper_red)
    # 返回像素值不为零的数(即白色像素点个数)
    red_pixels = cv2.countNonZero(red_mask)
    # 计算相应颜色像素在图片中的比例
    red_rate = red_pixels / (image.shape[0] * image.shape[1])
    print("red_rate:%f" % red_rate)

    # get yellow mask
    yellow_mask = cv2.inRange(hsv_model, lower_yellow, upper_yellow)
    # 返回像素值不为零的数(即白色像素点个数)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    # 计算相应颜色像素在图片中的比例
    yellow_rate = yellow_pixels / (image.shape[0] * image.shape[1])
    print("yellow_rate:%f" % yellow_rate)

    # get purple mask
    purple_mask = cv2.inRange(hsv_model, lower_purple, upper_purple)
    # 返回像素值不为零的数(即白色像素点个数)
    purple_pixels = cv2.countNonZero(purple_mask)
    # 计算相应颜色像素在图片中的比例
    purple_rate = purple_pixels / (image.shape[0] * image.shape[1])
    print("purple_rate:%f" % purple_rate)

    if ((red_rate > 0.1) or (yellow_rate > 0.1) or (purple_rate > 0.1)):
        color_dict = {"red": red_rate, "yellow": yellow_rate, "purple": purple_rate}
        # print(max(color_dict, key=color_dict.get))
        color = max(color_dict, key=color_dict.get)

        if color == "red":
            ret, thresh = cv2.threshold(red_mask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            """
            # 寻找二值图像的轮廓
            contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[1:6]
            cv2.drawContours(image, cnt, -1, (0, 0, 0), 2)
            """

            cv2.putText(image, color, (1, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        elif color == "yellow":
            ret, thresh = cv2.threshold(yellow_mask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            """
            # 寻找二值图像的轮廓
            contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[1:6]
            cv2.drawContours(image, cnt, -1, (0, 0, 0), 2)
            """
            cv2.putText(image, color, (1, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)

        elif color == "purple":
            ret, thresh = cv2.threshold(purple_mask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            """
            # 寻找二值图像的轮廓
            contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[1:6]
            cv2.drawContours(image, cnt, -1, (0, 0, 0), 2)
            """
            cv2.putText(image, color, (1, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)


if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    capture.set(3, 1280)
    capture.set(4, 720)

    while True:
        ret, frame = capture.read()
        color_detect(frame)

        cv2.namedWindow("frame", 0)
        cv2.resizeWindow("frame", 640, 360)
        cv2.imshow("frame", frame)
        out.write(frame)

        if cv2.waitKey(1) == 27:
            break

    capture.release()
    cv2.destroyAllWindows()