# common utils function of all python file



import pandas
# load the data from log file
# enter the wanted file path here
file_path = "logs/linear_example.log"
# file_path = "logs/random_example.log"
# file_path = "logs/application24_11_05_10_26_17.log"
def read_data(file_path):
    data_log = pandas.read_csv("../" + file_path, sep=";")
    bx = data_log["Bx"]
    by = data_log["By"]
    px = data_log["Px"]
    py = data_log["Py"]
    mdx = data_log["Mdx"]
    mdy = data_log["Mdy"]
    score = data_log["Score"]
    time = data_log["Time"]
    index = [i for i in range(len(mdx))]

    return bx, by, px, py, mdx, mdy, score, time

dbx = [ bx[i+1] - bx[i] for i in range(len(bx)-1) ]


if __name__ == "__main__":

    file_path = "logs/linear_example.log"

    res = read_data(file_path)