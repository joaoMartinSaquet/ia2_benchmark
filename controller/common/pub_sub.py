import time
import zmq
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import re


PLOT = True

topic_sub = b"GameData/"
topic_pub = b"Controller/"
server = "tcp://127.0.0.1"
game_data_socket_port = "5556"

controller_socket_port = "5560"
class Subscriber:

    """
    Class Subscriber

    This class is used to subscribe to a publisher and receive data from it.
    The data is stored in the class and can be accessed through the class methods.
    The class also has a method to plot the received data.
    """

    def __init__(self, plot=True):

        """
        Constructor of the Subscriber class
        
        Parameters
        ----------
        plot : bool
            If true, the class will plot the received data
        """

        self.plot = plot
        self.frame_counter = 0
        #  Socket to talk to subscriber
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.connect(server + ":" + game_data_socket_port)
        self.socket.setsockopt(zmq.CONFLATE, 1)

        # set Subscribe 
        # "" we subscirbe to every message
        # "x" we only subscirbe to messages that start with "x"
        # self.socket.setsockopt(zmq.SUBSCRIBE, topic_sub)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        if self.plot:
            self.fig, self.axs = plt.subplots(2,2)
            self.line_bx,  = self.axs[0,0].plot([],[], 'r--', label='ball x')
            self.line_px,  = self.axs[0,0].plot([],[], 'g', label='player x')
            self.line_score,  = self.axs[0,1].plot([],[], 'k-', label='ball x')
            self.line_mdx, = self.axs[1,0].plot([],[], 'r--', label='mouse dx')
            # line_mdy, = axs[1,0].plot([],[], 'g', label='mouse dy')

            # line_mdx, = axs[1,1].plot([],[], 'r--', label='mouse dx')
            self.line_mdy, = self.axs[1,1].plot([],[], 'orange', label='mouse dy')
                # Setting up plot labels and grids
            for ax_row in self.axs:
                for ax in ax_row:
                    ax.grid(True)
                    ax.legend()
    
        self.xs = []
        self.ys = []
        self.ts = []
        self.bxs = []
        self.bys = []
        self.pxs = []
        self.pys = []
        self.mdxs = []
        self.mdys = []
        self.scores = []

    def init(self):
        """
        Method to initialize the plot
        """
        self.line_bx.set_data([],[])
        self.line_px.set_data([],[])
        self.line_score.set_data([],[])
        self.line_mdx.set_data([],[])
        self.line_mdy.set_data([],[])
        return self.line_bx, self.line_px, self.line_score, self.line_mdx, self.line_mdy

    def update(self, frame):
        """
        Method to update the plot with new data
        """
        self.listen()
        self.line_bx.set_data(self.ts, self.bxs)
        self.line_px.set_data(self.ts, self.pxs)
        self.line_mdx.set_data(self.ts, self.mdxs)
        self.line_mdy.set_data(self.ts, self.mdys)
        self.line_score.set_data(self.ts, self.scores)

        # Rescale axes
        for ax in self.axs.flatten():
            ax.relim()
            ax.autoscale_view()
            
        return self.line_bx, self.line_px, self.line_score, self.line_mdx, self.line_mdy

    def listen(self):
        """
        Method to receive data from the publisher
        """
        data = self.socket.recv_multipart()
        # print("data : {}".format(data)) 
        topic, data = data[0], data[1]

        # print(f"Received request: {message}")
        matches = re.findall(r"(\w+)\s*:\s*(-?\d+\.\d+);?", data.decode("utf-8"))
        
        variables = {}
        # Convert matches to a dictionary with float values
        for var, val in matches:
            variables[var] = float(val)

        # Access each variable individually if needed
        self.bxs.append(variables["bx"])
        self.bys.append(variables["by"])
        self.pxs.append(variables["px"])
        self.pys.append(variables["py"])
        self.mdxs.append(variables["mdx"])
        self.mdys.append(variables["mdy"])
        self.scores.append(variables["score"])
        self.ts.append(variables["t"])

    def main(self):
        """
        Main method of the class
        """
        if self.plot:
            anim = FuncAnimation(self.fig, self.update, init_func=self.init, interval=1, blit=True)
            plt.show()
        else: 
            while True:
                self.listen()
                print("len received data : {}".format(len(self.bxs)))       

class Publisher:
    """
    Class to publish messages
    """
    def __init__(self):
        """
        Constructor of the Publisher class
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.connect(server + ":" + controller_socket_port)
        self.socket.setsockopt(zmq.CONFLATE, 1)

    def send(self, msg):
        """
        Method to send a message to the IA2 server
        """
        self.socket.send_string("%s, %s" % (topic_pub, msg))

if __name__ == "__main__":

    print("----------- Start IA2 Benchmark server -----------")
    subscriber = Subscriber(plot=PLOT)
    subscriber.main()

