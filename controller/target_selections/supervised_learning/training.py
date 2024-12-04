
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.preprocessing as preproc 


from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

import matplotlib.pyplot as plt

DEBUG = True 
if not DEBUG:
# Add the parent directory (two levels up) to sys.path
    sys.path.append(os.path.abspath(os.path.join('..', '..')))
    sys.path.append(os.path.abspath(os.path.join('..', '..', '..')))
    import common.tools as tools
    from utils import *
    LOG_FILE_PATH = "Dataset/targt_selection_normal_interactions.log"

else: 
    sys.path.append(os.path.abspath(os.path.join('')))
    import controller.common.tools as tools
    LOG_FILE_PATH = "controller/target_selections/supervised_learning/Dataset/longer_stay_on_target.log"

    

import numpy as np
import matplotlib.pyplot as plt
from utils import *
import argparse


INDEX = -1
STACK = False
MAX = 34

class ANN(nn.Module):

    def __init__(self, neurons = [2, 128, 128, 2]):
        super(ANN, self).__init__()

        self.layer = nn.ModuleList([nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))])

    def forward(self, x):

        for i in range(len(self.layer)):
            x = self.layer[i](x)
            if i != len(self.layer) - 1:
                x = nn.ReLU()(x)

        return x



class datasetTarget(Dataset):
    def __init__(self, x, y):
        self.mouse_movement = x
        self.best_dir = y

    def __len__(self):
        return len(self.mouse_movement)

    def __getitem__(self, index):
        return self.mouse_movement[index], self.best_dir[index]
    

def training(args):

    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"  


    # training parameters
    epochs = 100
    lr = 5e-4


    # load data 
    Tx, Ty, px, py, mdx, mdy, score, time, dbx, dby, dataframe = read_data(LOG_FILE_PATH)

    device_disp = np.vstack((mdx[:INDEX], mdy[:INDEX])).T


    scaler = preproc.MinMaxScaler().fit(device_disp)

    scaled_disp = scaler.transform(device_disp)

    # set target
    target_positions = np.vstack(( Tx[:INDEX], Ty[:INDEX])).T
    players_positions = np.vstack(( px[:INDEX], py[:INDEX])).T

    dir = (target_positions - players_positions)

    # To polaire
    ang = np.atan2(dir[:,1], dir[:,0])
    mag = np.sqrt(dir[:,0]**2 + dir[:,1]**2)

    # scale magnitude to max then maximum movement is not exceeded
    np.clip(mag, -0, MAX, out=mag)
    best_disp = np.vstack((mag * np.cos(ang), mag * np.sin(ang))).T
    scaled_best_disp = scaler.transform(best_disp)





    # without STACK

    # with stack 
    if STACK:
        scaled_mdx = scaled_disp[:,0]
        scaled_mdy = scaled_disp[:,1]
        n_stack = 50
        xfeature = torch.tensor(np.hstack((np.array(tools.stack(scaled_mdx, n_stack=n_stack)),
                                            np.array(tools.stack(scaled_mdy,  n_stack=n_stack))))).to(torch.float32).to(dev)

        in_feature = 2 * n_stack
    else:
        xfeature = torch.tensor(scaled_disp).to(torch.float32).to(dev) #scaled_disp 
        in_feature = 2

    ytarget = torch.tensor(scaled_best_disp).to(torch.float32).to(dev) #scaled_best_disp
    # ytarget = torch.tensor(scaled_disp).to(torch.float32).to(dev) #scaled_dis



    # print("target_positions shape ", target_positions.shape)
    # print("players_positions shape ", players_positions.shape)
    # print("y shape ", y.shape)

    # train the model 
    if args.model == "ann":
        ann_controller = ANN([in_feature, 128, 2]).to(dev)
        opt = torch.optim.Adam(ann_controller.parameters(), lr=lr)
    dataset = datasetTarget(xfeature, ytarget)

    # come from the fact that 
    train, val = random_split(dataset, [0.9, 0.1])

    trainloader = DataLoader(train, batch_size=128, shuffle=True)   
    valloader = DataLoader(val, batch_size=128, shuffle=True)
    
    loss_val = []
    validation_loss = []
    criterion = nn.MSELoss()
    for epoch in range(epochs):

        running_loss = 0.
        # train one epochs
        for i, data in enumerate(trainloader):
            x, y = data

            # free gradient before each bacth
            opt.zero_grad()
            # TODO find a better loss ! true disp might not be able to get the traj !

            y_pred = ann_controller(x)

            # MSE on vector directly 
            loss = criterion(y, y_pred)

            # MSE get co dir vector
            
            loss.backward()
            opt.step()
            # opt.zero_grad()

            # Gather data and report
            running_loss += loss.item()
        loss_val.append(running_loss/i)

        last_loss = 0
        with torch.no_grad():
            for j ,data in enumerate(valloader):
                x, y = data
                y_pred = ann_controller(x)

                # loss in polar
                loss = nn.MSELoss()(y, y_pred) 
                last_loss += loss.item() 
            validation_loss.append(last_loss)

        if epoch % 10  == 0 or epoch == epochs:
            print("epochs : {} training loss : {}, Validation loss : {}".format(epoch, loss_val[-1], validation_loss[-1]))
    
    fig, axs = plt.subplots(3,1)
    axs[0].plot(loss_val, 'b', label='training loss')
    axs[0].plot(validation_loss, 'purple', label='validation loss')
    axs[0].set_title("loss")
    axs[0].legend()

    # axs[1].plot(validation_loss, 'purple', label='validation loss')

    n_pt, controller_d = run_episodes(ann_controller, xfeature, time.to_list()[:INDEX], scaler=scaler)
    oracle_pt, dir_pt = run_oracle(target_positions, time.to_list()[:INDEX])

    ax3d1 = plt.figure().add_subplot(projection='3d')
    # ax3d1.plot(time.to_list()[:INDEX], n_pt[:,0], n_pt[:,1], 'b', label='point')
    ax3d1.plot(time.to_list()[:INDEX], oracle_pt[:,0], oracle_pt[:,1], 'g', label='oracle')
    ax3d1.plot(time.to_list()[:INDEX], target_positions[:,0], target_positions[:,1], 'y.', label='oracle')

    ax3d1.legend()
    ax3d1.set_xlabel('time')
    ax3d1.set_ylabel('x')
    ax3d1.set_zlabel('y')
    ax3d1.view_init(elev=0., azim=0, roll=0)

    ax3d2 = plt.figure().add_subplot(projection='3d')
    ax3d2.plot(time.to_list(), mdx, mdy, '.b', label='device ')
    ax3d2.plot(time.to_list()[:INDEX], controller_d[:,0], controller_d[:,1], 'r.', label='IA controller')
    ax3d2.plot(time.to_list()[:INDEX], best_disp[:, 0], best_disp[:, 1], 'g.', label='true ')
    
    ax3d2.legend()
    ax3d2.set_xlabel('time')
    ax3d2.set_ylabel('mdx')
    ax3d2.set_zlabel('mdy')
    ax3d2.view_init(elev=0., azim=0, roll=0)

    # axs[1].set_xbound(-1000, 1000)
    # axs[1].set_ybound(-1000, 1000)

    axs[2].plot(mdx, mdy, 'b.', label = 'mouse')
    axs[2].plot(controller_d[:,0], controller_d[:,1], 'r.', label = 'controller')
    axs[2].plot(best_disp[:,0], best_disp[:,1], 'g.', label = 'true dir ')
    
    axs[2].legend()
    plt.show()

    
def run_oracle(target, time):
    player_position_x = 0  # player piosition always start at 0
    player_position_y = 0
    Pt = []
    best_disp = []
    with torch.no_grad():
        for i, t in enumerate(time):
            
            dir = target[i, :] - np.array([player_position_x, player_position_y])
            # scale dir to get realistic displacement
            # To polaire
            ang = np.atan2(dir[1], dir[0])
            mag = np.sqrt(dir[0]**2 + dir[1]**2)

            # scale magnitude to max then maximum movement is not exceeded
            if mag > MAX:
                mag = MAX

            oracle_disp = np.vstack((mag * np.cos(ang), mag * np.sin(ang))).T


            # update player position
            player_position_x += oracle_disp[0][0].item()
            player_position_y += oracle_disp[0][1].item()
            best_disp.append([oracle_disp[0][0].item(), oracle_disp[0][1].item()])

            # print("player position at time t : {} : ({}, {})".format(t, player_position_x, player_position_y))
            # update target position
            Pt.append([player_position_x, player_position_y])


    return np.array(Pt), np.array(best_disp)


def run_episodes(controller, cursor_d, time, scaler = None):

    # epiose run
    player_position_x = 0  # player piosition always start at 0
    player_position_y = 0
    Pt = []
    device_d = []
    with torch.no_grad():
        for i, t in enumerate(time):
            
            # get the player displacement
            mouse_disp = cursor_d[i, :]


            # compute the controler estimated displacement
            controller_disp = controller(mouse_disp)

            if scaler is not None:
               unscaled_disp = scaler.inverse_transform(controller_disp.cpu().numpy().reshape(1,-1))
            else :
                unscaled_disp = controller_disp.cpu().numpy().reshape(1,-1)
            # update player position
            player_position_x += unscaled_disp[0][0].item()
            player_position_y += unscaled_disp[0][1].item()
            device_d.append([unscaled_disp[0][0].item(), unscaled_disp[0][1].item()])

            # print("player position at time t : {} : ({}, {})".format(t, player_position_x, player_position_y))
            # update target position
            Pt.append([player_position_x, player_position_y])


    return np.array(Pt), np.array(device_d)




if __name__ == "__main__":

    print("----------- training -----------")
    # Create the argument parser
    parser = argparse.ArgumentParser(description="A script to demonstrate argument parsing.")
    
    # Add arguments
    parser.add_argument("--model", type=str, default="ann", help="model to train")
    parser.add_argument("--log", type=bool, default=False, help="dumping log or not")
    parser.add_argument("--normalize", type=str, default="..", help="path to log")
    # parser.add_argument("--greet", action="store_true", help="Include this flag to print a greeting")
    
    # Parse the arguments
    args = parser.parse_args()

    training(args)


