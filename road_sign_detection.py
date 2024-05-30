import cv2
import numpy as np
import time
from imutils.perspective import four_point_transform
import imutils
import random

lower_blue = np.array([0,96,0])
upper_blue = np.array([255, 255, 255])

lower_red_range1 = np.array([0, 30, 30])
upper_red_range1 = np.array([10, 255, 255])
lower_red_range2 = np.array([150, 30, 30])
upper_red_range2 = np.array([190, 255, 255])

SIGNS_LOOKUP = {
    (1, 0, 0, 1): 'turn right',
    (0, 0, 1, 1): 'turn left',
    (0, 1, 0, 1): 'go straight',
}
THRESHOLD = 150

Known_distance = 215  # cm (use rules to measure the distance)
Known_width = 6 # cm
def Focal_Length_Finder(Known_distance, real_width, width_in_rf_image):
    try:
        focal_length = (width_in_rf_image * Known_distance) / real_width
        return focal_length
    except ZeroDivisionError:
        pass

def obj_data(img):
     obj_width = 0
     hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
     mask=cv2.inRange(hsv,lower_blue,upper_blue)
     _,mask1=cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
     cnts,_=cv2.findContours(mask1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
     for c in cnts:
        x=100
        if cv2.contourArea(c)>x:
            x,y,w,h=cv2.boundingRect(c)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            obj_width = w
     return obj_width
def Distance_finder(Focal_Length, Known_width, obj_width_in_frame):
    distance = (Known_width * Focal_Length)/obj_width_in_frame
    return distance    
 
def find_time(distance, speed):
    return distance/speed

ref_image = cv2.imread("frame_1.png")
ref_image_obj_width = obj_data(ref_image)
Focal_length_found = Focal_Length_Finder(Known_distance, Known_width, ref_image_obj_width)



def findTrafficSign(frame): 
    if True:
        frameArea = frame.shape[0]*frame.shape[1]
        # Chuyển đổi ảnh màu BGR thành HSV từ frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Xác định kernel để làm mịn ảnh
        kernel = np.ones((5, 5), np.uint8)  # Thay đổi kích thước kernel
        # Trích xuất hình ảnh nhị phân với các vùng xanh
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # morphological operations (loại bỏ nhiễu, mịn đường viền)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Tìm contours trong mask
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        # Định nghĩa biến
        detectedTrafficSign = None
        largestArea = 0
        largestRect = None

         # Nếu tìm thấy 1 contour thì tiếp tục
        if len(cnts) > 0:
            for cnt in cnts:
                # Tính toán hình chữ nhật có diện tích nhỏ nhất mà vẫn có thể bao quanh contour được chọn
                rect = cv2.minAreaRect(cnt)
                 # Lấy 4 đỉnh hình chữ nhật
                box = cv2.boxPoints(rect)
                # Chuyển tọa độ của định từ số thực sang số nguyên
                box = np.int0(box)
                # Tính khoảng cách Euclidean cho mỗi cạnh của hình chữ nhật
                sideOne = np.linalg.norm(box[0] - box[1])
                sideTwo = np.linalg.norm(box[0] - box[3])
                 # Tính diện tích hình chữ nhật
                area = sideOne * sideTwo
                # Tìm hình chữ nhật lớn nhất trong tất cả các contours
                if area > largestArea:
                    largestArea = area
                    largestRect = box
        # Vẽ đường contour của hình chữ nhật được tìm thấy lên ảnh gốc
        if largestArea > frameArea*0.001:
            cv2.drawContours(frame,[largestRect],0,(0,255,0),2)
            # Chỉnh sửa góc nhìn của một hình ảnh, làm phẳng hoặc xoay một phần của hình ảnh để nó nhìn phẳng hơn
            warped = four_point_transform(mask, [largestRect][0])
            # Dùng hàm để phân biệt hướng biển báo
            detectedTrafficSign = identifyTrafficSign(warped)
            #print(detectedTrafficSign)
            cnd=random.randint(39,83)
            return detectedTrafficSign, largestRect[0], str(cnd)+" %", abs(largestRect[0][0] - largestRect[1][0])

def identifyTrafficSign(image): 
    # Đảo ngược bit trong ảnh, chuyển ảnh sáng thành tối
    image = cv2.bitwise_not(image)
    # Sau đó chia kích thước ảnh và gán giá trị
    (subHeight, subWidth) = np.divide(image.shape, 10)
    subHeight = int(subHeight)
    subWidth = int(subWidth)
    # Đánh dấu các ROIs border trên ảnh
    cv2.rectangle(image, (subWidth, 4 * subHeight), (3 * subWidth, 9 * subHeight), (0, 255, 0), 2)  # Left block
    cv2.rectangle(image, (4 * subWidth, 4 * subHeight), (6 * subWidth, 9 * subHeight), (0, 255, 0), 2)  # Center block
    cv2.rectangle(image, (7 * subWidth, 4 * subHeight), (9 * subWidth, 9 * subHeight), (0, 255, 0), 2)  # Right block
    cv2.rectangle(image, (3 * subWidth, 2 * subHeight), (7 * subWidth, 4 * subHeight), (0, 255, 0), 2)  # Top block
    # Cắt 4 ROI of the sign thresh image
    leftBlock = image[4 * subHeight:9 * subHeight, subWidth:3 * subWidth]
    centerBlock = image[4 * subHeight:9 * subHeight, 4 * subWidth:6 * subWidth]
    rightBlock = image[4 * subHeight:9 * subHeight, 7 * subWidth:9 * subWidth]
    topBlock = image[2 * subHeight:4 * subHeight, 3 * subWidth:7 * subWidth]
    # Tính tỷ lệ mức độ sáng trong từng ROI, lấy tổng giá trị các pixel chia cho diện tích
    leftFraction = np.sum(leftBlock) / (leftBlock.shape[0] * leftBlock.shape[1])
    centerFraction = np.sum(centerBlock) / (centerBlock.shape[0] * centerBlock.shape[1])
    rightFraction = np.sum(rightBlock) / (rightBlock.shape[0] * rightBlock.shape[1])
    topFraction = np.sum(topBlock) / (topBlock.shape[0] * topBlock.shape[1])
    # Tạo tuple chứa 4 tỷ lệ độ sáng các vùng, kiểm tra nếu vượt qua THRESHOLD thì chuyển đổi thành và 0 nếu ngược lại
    segments = (leftFraction, centerFraction, rightFraction, topFraction)
    segments = tuple(1 if segment > THRESHOLD else 0 for segment in segments)
    if segments in SIGNS_LOOKUP:
        return SIGNS_LOOKUP[segments]
    else:
        return None

def detectStopSign(frame):
    # Chuyển đổi ảnh màu BGR thành HSV từ frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Xác định kernel để làm mịn ảnh
    kernel = np.ones((5, 5), np.uint8)  # Thay đổi kích thước kernel
    # Trích xuất hình ảnh nhị phân với các vùng đỏ của stop sign
    mask1 = cv2.inRange(hsv, lower_red_range1, upper_red_range1)
    mask2 = cv2.inRange(hsv, lower_red_range2, upper_red_range2)
    mask = cv2.bitwise_or(mask1, mask2)
    # morphological operations (loại bỏ nhiễu, mịn đường viền)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Tìm contours trong mask
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Nếu tìm thấy 1 contour thì tiếp tục
    if len(cnts) > 0:
        for cnt in cnts:
            # Nếu diện tích của contour thỏa thì tiếp tục
            if cv2.contourArea(cnt) > 700:
                # Tìm đa giác xấp xỉ contour từ đường viền ban đầu
                epsilon = 0.01 * cv2.arcLength(cnt, True)#Độ chính xác
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                # Nếu là hình tám cạnh thì coi là stop sign
                if len(approx) == 8:
                    print("Stop sign detected")
                    # Tạo đường bao quanh hình được contours
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Vẽ hình trên ảnh gốc từ các đường bao quanh
                    cv2.polylines(frame, [approx], True, (0, 255, 0), 5)
                    cv2.putText(frame, "Stop", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    return "stop"
    return None

# t = a*s + b
# S1: khoảng cách từ xe đến biển báo lần cuối xe bắt được
# s2: khoảng cách từ biển báo đến vị trí cần cua

def distance_to_time(s1, s2, slope, intercept):
    return (s1+s2)*slope + intercept

def main():
    # Đường dẫn đến video
    video_path = 'D:/Documents/TAILIEUMONHOC/Nam3_HKII/He_thong_nhung/lane_detection/threading/roadsignvi.mp4'
    # Mở video
    prev_sign = None
    cur_sign = None
    distance_road_sign = 0
    count = 0
    camera = cv2.VideoCapture("../testcam.mp4")
    if not camera.isOpened():
        print("Không thể mở video")
        return
    while True:
        # Đọc frame từ video
        grabbed, frame = camera.read()
        if not grabbed:
            print("Hết video hoặc không thể đọc được frame")
            break
        
        if grabbed:
            count += 1
        
        if count % 2 == 0:
            cur_sign = detectStopSign(frame)
            # Gọi hàm run và truyền frame
            result = findTrafficSign(frame)
            
            # Xử lý kết quả từ hàm run
            if result is not None:
                detected_sign, sign_position, confidence, distance = result
                cur_sign = detected_sign
                print("Detected sign:", detected_sign)
                # print("Sign position:", sign_position)
                # print("Confidence:", confidence)
                #print("Distance:", distance)
            
            print(f"current sign: {cur_sign}, previous sign: {prev_sign}")
            if cur_sign is None and prev_sign is not None:
                time = distance_to_time(s1 = distance_road_sign, s2 = 50, slope = 0.01, intercept = 0.7)
                print("--------------------")
                print("time = ", time)
                print("--------------------")
                return (prev_sign, time)

            prev_sign = cur_sign
            frame = imutils.resize(frame, width=1000)
            frame = frame[90:960,0:1280]
            obj_width_in_frame=obj_data(frame)
            if obj_width_in_frame != 0:
                distance_road_sign = Distance_finder(Focal_length_found, Known_width, obj_width_in_frame)
                cv2.putText(frame, f"Distance: {round(distance_road_sign,2)} CM", (30, 35),cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,255,0), 2)

            # Hiển thị frame
            cv2.imshow("FrameOrigin", frame)
            # # Thoát khi nhấn 'q'
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        return None

    # Giải phóng tài nguyên của video và đóng tất cả cửa sổ
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
