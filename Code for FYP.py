import rospy
import geometry_msgs.msg
import threading
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import cv2
from quest2 import touchstate

drive_para = geometry_msgs.msg.Twist()
pub = rospy.Publisher('/cmd_vel', geometry_msgs.msg.Twist, queue_size=10)

latest_cv_img = None
latest_depth_img = None
safe=0
model=YOLO("/home/haiyi/fyp/runs/detect/train/weights/best.pt")

def vins_img_callback(data):
    global latest_cv_img
    try:
        cv_img = bridge.imgmsg_to_cv2(data, "bgr8")
        latest_cv_img = cv_img
    except CvBridgeError as e:
        print(e)

def vins_dep_callback(data):
    global latest_depth_img
    try:
        depth_img = bridge.imgmsg_to_cv2(data, "16UC1")
        latest_depth_img = depth_img
    except CvBridgeError as e:
        print(e)

def release():
    global drive_para
    while not rospy.is_shutdown():
        pub.publish(drive_para)
        rospy.loginfo(drive_para)

class Control():
    def __init__(self):
        super().__init__()

    def forward(self):
        global drive_para
        drive_para.linear.x = 0.5
        drive_para.angular.z = 0
        
    def back(self):
        global drive_para
        drive_para.linear.x = -0.5
        drive_para.angular.z = 0

    def left(self):
        global drive_para
        drive_para.linear.x = 0
        drive_para.angular.z = -0.4

    def right(self):
        global drive_para
        drive_para.linear.x = 0
        drive_para.angular.z = 0.4
        
    def stop(self):
        global drive_para
        drive_para.linear.x = 0
        drive_para.angular.z = 0

def safecheck(numbers):
    for num in numbers:
        if num <= 500:
            return 0
    return 1

ctl = Control()
ts=touchstate()
#ts.get() return 1-forward 2-back 3-left 4-right
def control():
    global safe
    if safe==1:
        cmd=ts.get()
        if cmd==1:
            ctl.forward()
        elif cmd==2:
            ctl.back()
        elif cmd==3:
            ctl.left()
        elif cmd==4:
            ctl.right()
    else:
        ctl.stop()
        
if __name__ == '__main__':
    rospy.init_node('driver')   
    t = threading.Thread(target=release)
    t.setDaemon(True)  # Set the thread as a daemon thread
    t.start()
    t2 = threading.Thread(target=control)
    t2.setDaemon(True)  # Set the thread as a daemon thread
    t2.start()
    
    bridge = CvBridge()
    rospy.Subscriber("/camera/color/image_raw", Image, vins_img_callback, queue_size=1)
    rospy.Subscriber("/camera/depth/image_raw", Image, vins_dep_callback, queue_size=1)
    
    while not rospy.is_shutdown():
        if latest_cv_img is not None and latest_depth_img is not None:
            # Process the latest_cv_img and latest_depth_img
            results = model.predict(source=latest_cv_img, save=True, save_txt=True)
            boxes=results[0][0].boxes
            depths=[]
            for i in range(len(boxes)):
                box = boxes[i]  # returns one box
                if box.conf>0.8:
                    x=box.xywh[1]+box.xywh[3]*0.5
                    y=box.xywh[2]+box.xywh[4]*0.5
                    depth=latest_depth_img[x,y]
                    depths.append(depth)
                    text = str(depth/1000)+"m"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    position = (x,y)
                    font_scale = 1
                    font_color = (255, 255, 255)
                    font_thickness = 2
                    cv2.putText(latest_cv_img, text, position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            safe=safecheck(depths)  
            cv2.imshow("Image with Text", latest_cv_img)            
        rospy.sleep(0.1)