# common utils function of all python file



import pandas
# load the data from log file
# enter the wanted file path here
file_path = "logs/linear_example.log"
# file_path = "logs/random_example.log"
# file_path = "logs/application24_11_05_10_26_17.log"
def read_data(file_path):
    """
    Reads a log file and extracts various columns of data. Computes the differences
    between consecutive elements in the Bx and By columns.

    Parameters:
    file_path (str): The path to the log file.

    Returns:
    tuple: Contains the following elements:
        - bx (pandas.Series): Series of Bx values.
        - by (pandas.Series): Series of By values.
        - px (pandas.Series): Series of Px values.
        - py (pandas.Series): Series of Py values.
        - mdx (pandas.Series): Series of Mdx values.
        - mdy (pandas.Series): Series of Mdy values.
        - score (pandas.Series): Series of Score values.
        - time (pandas.Series): Series of Time values.
        - dbx (list): List of differences between consecutive Bx values.
        - dby (list): List of differences between consecutive By values.
    """
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
    dbx = [bx[i+1] - bx[i] for i in range(len(bx)-1)]
    dby = [by[i+1] - by[i] for i in range(len(by)-1)]

    return bx, by, px, py, mdx, mdy, score, time, dbx, dby



if __name__ == "__main__":

    file_path = "logs/linear_example.log"

    res = read_data(file_path)