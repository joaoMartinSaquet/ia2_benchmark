from os.path import dirname, abspath
d = dirname(dirname(abspath(__file__)))
import sys
sys.path.append(d)

from common.pub_sub import Subscriber, Publisher

from collections import deque

class controller():
    
    def __init__(self):
        
        self.subscriber = Subscriber()
        self.publisher = Publisher()

        self.dx_ctrl = deque(maxlen=2)
        self.dy_ctrl = deque(maxlen=2)
        

    def main(self):

        while True:

            self.subscriber.listen()

            # do oracle routines
            self.dx_ctrl.append(self.subscriber.bxs[-1])
            self.dy_ctrl.append(self.subscriber.bys[-1])
            if len(self.dx_ctrl) > 1:
                hdx = self.dx_ctrl[0] - self.dx_ctrl[1]
                hdy = self.dy_ctrl[0] - self.dy_ctrl[1]

                msg = f"mdx: {hdx}; mdy: {hdy};"
                print("send : {}".format(msg))
                self.publisher.send(msg)
        


            
                

    

if __name__ == "__main__":

    print("----------- Start IA2 Benchmark server -----------")

    controller = controller()

    controller.main()

