import matplotlib
matplotlib.use('TkAgg')

import  matplotlib.pyplot as plt


def draw(loss, trRSME, valRSME):
    plt.subplot(2, 1, 1)
    x = range(len(loss))

    plt.plot(x, loss, 'g-')
    plt.title('Train Loss')

    plt.subplot(2, 1, 2)

    plt.plot(x, trRSME, 'b-', x, valRSME, 'g-')
    plt.title('Train(-Blue-) RSME, and Validation(-Green-) RSME')

    plt.show()
