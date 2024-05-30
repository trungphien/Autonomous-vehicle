import threading
import time
import keyboard
# sq

stop_thread_flag = False

# Hàm lane + điều khiển
def lane_detection_thread():
    while not stop_thread_flag:
        print("Lane detection in progress...")
        # online_detect.main()  # code lane và góc lái
        time.sleep(1) 

def road_signs_detection_thread():
    while not stop_thread_flag:
        print("Road signs detection in progress...")
        # road_sign_detect.main()  # Trả về thời gian cua + tên biển báo
        time.sleep(1)

def control():
    while not stop_thread_flag:
        print("Control in progress...")

def start_threads():
    global stop_thread_flag
    stop_thread_flag = False

    # Create threads for each module
    lane_thread = threading.Thread(target=lane_detection_thread, daemon=True)
    road_signs_thread = threading.Thread(target=road_signs_detection_thread, daemon=True)

    # Start both threads
    lane_thread.start()
    road_signs_thread.start()
    # Wait for both threads to finish
    # lane_thread.join()
    # road_signs_thread.join()

def stop_threads():
    global stop_thread_flag
    stop_thread_flag = True

if __name__ == "__main__":
    
    print("Press 's' to start threads and 'q' to stop threads")
    keyboard.add_hotkey('s', start_threads)  # Start threads when 's' is pressed
    # keyboard.add_hotkey('q', stop_threads)  # Stop threads when 'q' is pressed
    keyboard.wait()  # Wait for keyboard input