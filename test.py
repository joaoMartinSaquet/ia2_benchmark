import pyautogui
import time
from collections import deque


x_pos = deque(maxlen=2)
y_pos = deque(maxlen=2)

while True:

    x, y = pyautogui.position()
    x_pos.append(x)
    y_pos.append(y)
    time.sleep(0.01)

    

    if len(x_pos) > 1: 
        print("mouse dx : ", x_pos[1] - x_pos[0])
        # print("mouse dy : ", y_pos[1] - y_pos[0])