from collections import deque
import numpy as np

def write_to_file(file_name, dx, dy):

    with open(file_name, 'w') as f:
        f.write("dx;dy\n")
        for i in range(len(dx)):
            f.write(f"{dx[i]};{dy[i]}\n")


def stack(x, n_stack = 4):
    """gather x into J stack of n_stack

    Args:
        x (_type_): _description_

    Tests
        a_stack = stack(a, 2)

        index = 3

        print("stack a at index {} : \n {}".format(index, a_stack[index]))
        print("a at index {} : \n {}".format(index, a[index:(index+1)]))

        
    """
    stacked_x = deque([], maxlen = n_stack)
    # init stacked_x 
    for i in range(n_stack):
        stacked_x.append(0.0)
    
    vec_stack = []
    for i in range(len(x)):

        # we add x to the stack
        stacked_x.append(x[i])

        # we add the stack to the output vec
        vec_stack.append(np.array(list(stacked_x)))

    return  vec_stack

