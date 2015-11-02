import random

def shuffle(x,y,iter=20):
    for i in range(len(x) * iter):
        rand1 = random.random() * len(x)
        rand2 = random.random() * len(x)
        while rand1 != rand2:
            rand2 = random.random() * len(x)

        temp_x = x[rand1]
        x[rand1] = x[rand2]
        x[rand2] = temp_x

        temp_y = y[rand1]
        y[rand1] = y[rand2]
        y[rand2] = temp_y