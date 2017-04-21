import matplotlib
matplotlib.use('TkAgg')

from functools import wraps
import  matplotlib.pyplot as plt


def track_plot(func):
    plt.ion()
    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.called += 1

        loss, trRMSE, valRSME = func(*args, **kwargs)

        wrapper.loss.append(loss)
        wrapper.trRSME.append(trRMSE)
        wrapper.valRSME.append(valRSME)

        plt.clf()
        plt.subplot(2, 1, 1)
        x = range(wrapper.called)

        plt.plot(x, wrapper.loss, 'g-')
        plt.title('Train Loss')

        plt.subplot(2, 1, 2)

        plt.plot(x, wrapper.trRSME, 'b-', x, wrapper.valRSME, 'g-')
        plt.title('Train(-Blue-) RSME, and Validation(-Green-) RSME')

        plt.pause(0.00001)
        plt.show(block=False)

        return loss, trRMSE, valRSME

    wrapper.called = 0
    wrapper.loss = []
    wrapper.trRSME = []
    wrapper.valRSME = []

    wrapper.__name__ = func.__name__

    return wrapper


def draw(loss, trRSME, valRSME):
    plt.subplot(2, 1, 1)
    x = range(len(loss))

    plt.plot(x, loss, 'g-')
    plt.title('Train Loss')

    plt.subplot(2, 1, 2)

    plt.plot(x, trRSME, 'b-', x, valRSME, 'g-')
    plt.title('Train(-Blue-) RSME, and Validation(-Green-) RSME')

    plt.show()
