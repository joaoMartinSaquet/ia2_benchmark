import time
import zmq
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import re
from controller.common.pub_sub import Subscriber, Publisher


if __name__ == "__main__":

    print("----------- Start IA2 Benchmark Server -----------")


    # Create animation
    Subscriber = Subscriber(plot=True)
    Subscriber.main()



