import math

from djitellopy import *
import cv2
import time
from pynput import keyboard
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as pltq

# ARUCO MARKER SIDE LENGTH (meters)
aruco_marker_side_length = 0.1415

# TARGET POSITION FROM TAG (meters)
targetZ = 1
targetX = 0
targetY = 0

count = 0

width = 320
height = 240

def wait(seconds):
    time.sleep(seconds)

tello = Tello()
tello.connect()
tello.streamon()
frameRead = tello.get_frame_read()

tello.TIME_BTW_RC_CONTROL_COMMANDS = 0.1

rotSpeed = 22
speed = 50

def on_press(key):
    global count
    if key == keyboard.KeyCode(char='a'):
        tello.rotate_counter_clockwise(rotSpeed)
    elif key == keyboard.KeyCode(char='d'):
        tello.rotate_clockwise(rotSpeed)
    elif key == keyboard.KeyCode(char='w'):
        tello.move_forward(speed)
    elif key == keyboard.KeyCode(char='s'):
        tello.move_back(speed)
    elif key == keyboard.KeyCode(char='e'):
        cv2.imwrite(f'pictures/{count}.png', frameRead.frame)
        count += 1
    elif key == keyboard.Key.space and tello.is_flying != True:
        tello.takeoff()

print("tello initialized")

print("Battery: ", tello.get_battery())

corners, ids, rejected = None, None, None


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z

aruco_dict = None
parameters = None
mtx = None
dst = None

def initCV():
    global aruco_dict, parameters, mtx, dst
    # Define the dictionary and parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()

    cv_file = cv2.FileStorage(
        "calibration_chessboard.yaml", cv2.FILE_STORAGE_READ)
    mtx = cv_file.getNode('K').mat()
    dst = cv_file.getNode('D').mat()
    cv_file.release()

transform_translation_x = 0
transform_translation_y = 0
transform_translation_z = 0
yaw_z = 0

marker_ids = None

def detectTags(frame):
    global corners, ids, rejected, aruco_dict, parameters, aruco_marker_side_length, transform_translation_x, transform_translation_y, transform_translation_z, marker_ids, yaw_z
    if len(frame.shape) == 2:  # Grayscale image
        frame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:  # RGBA image
        frame = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create the ArUco detector
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Detect the markers
    (corners, marker_ids, rejected) = cv2.aruco.detectMarkers(
        gray, aruco_dict)
    # print("Detected markers:", ids)

    rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(
        corners,
        aruco_marker_side_length,
        mtx,
        dst)
    if marker_ids is not None:
        for i, marker_id in enumerate(marker_ids):
            # Store the translation (i.e. position) information
            transform_translation_x = tvecs[i][0][0]
            transform_translation_y = tvecs[i][0][1]
            transform_translation_z = tvecs[i][0][2]

            # Store the rotation information
            rotation_matrix = np.eye(4)
            rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
            r = R.from_matrix(rotation_matrix[0:3, 0:3])
            quat = r.as_quat()

            # Quaternion format
            transform_rotation_x = quat[0]
            transform_rotation_y = quat[1]
            transform_rotation_z = quat[2]
            transform_rotation_w = quat[3]

            # Euler angle format in radians
            roll_x, pitch_y, yaw_z = euler_from_quaternion(transform_rotation_x,
                                                           transform_rotation_y,
                                                           transform_rotation_z,
                                                           transform_rotation_w)

            roll_x = math.degrees(roll_x)
            pitch_y = math.degrees(pitch_y)
            yaw_z = math.degrees(yaw_z)
            # print("transform_translation_x: {}".format(transform_translation_x))
            print("transform_translation_y: {}".format(transform_translation_y))
            # print("transform_translation_z: {}".format(transform_translation_z))
            # print("roll_x: {}".format(roll_x))
            # print("pitch_y: {}".format(pitch_y))
            print("yaw_z: {}".format(yaw_z))
            # print()

            # Draw the axes on the marker
            # cv2.aruco.drawAxis(frame, mtx, dst, rvecs[i], tvecs[i], 0.05)


listener = keyboard.Listener(on_press=on_press)
listener.start()

initCV()

error = 0.05

def move():
    global targetX, targetY, targetZ, transform_translation_x, transform_translation_y, transform_translation_z, marker_ids, targetYaw

    zdif = targetZ - transform_translation_z
    xdif = targetX - transform_translation_x
    ydif = targetY - transform_translation_y

    speed = 100
    vertSpeed = 10
    rotSpeed = 100

    forward_back = 0
    left_right = 0
    up_down = 0
    yaw = 0
    if marker_ids is not None:
        forward_back = -int(zdif * speed)
        left_right = -int(xdif * speed )
        up_down = -int(ydif * vertSpeed)
        yaw = -int(xdif * rotSpeed)
    print("left-right: ", left_right)
    print("forward-back: ", forward_back)
    print("up-down: ", up_down)
    print("rotation: ", rotSpeed * transform_translation_x)

    if tello.is_flying:
        tello.send_rc_control(left_right, forward_back, up_down, yaw)

while True:
    # video handling
    frame = frameRead.frame
    detectTags(frame)
    if tello.is_flying:
        move()
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.imshow("Live View", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

if tello.is_flying:
    tello.land()

tello.streamoff()

