import cv2
import time
import os
import psutil

processId = os.getpid()
process = psutil.Process(processId)
base_memory = process.memory_info().rss

if __name__ == '__main__':
    print(process.memory_info().rss - base_memory)

    start = time.time()
    stream = cv2.VideoCapture(0)
    ret, image = stream.read()
    print(image.shape)
    # stream.set(cv2.CAP_PROP_FPS, 10)
    s = 0
    s_count = 0
    while True:
        ret, image = stream.read()
        cv2.imshow('', image)
        cv2.waitKey(1)
        s_count += 1
        s_temp = int(time.time() - start)
        if s_temp > s:
            s_count = 1
            print(s, (process.memory_info().rss - base_memory)/1024/1024)
        s = s_temp
        print(s, s_count)
