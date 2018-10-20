import numpy as np
import matplotlib.pyplot as plt

# use get_model function to get the regression model
# p means polynomial coefficients
def get_model(p, input_x):
    model = np.polyval(np.polyfit(x, y, p), input_x)
    return model

# return the difference for each set of p, x, y
def get_cost(p, input_x, input_y):
    cost = 0.5 * ((get_model(p, input_x) - input_y) ** 2).sum()
    return cost


# import data and transform it into numpy array
x, y = [], []
for sample in open("data.txt", "r"):
    _x, _y = sample.split(",")
    x.append(float(_x))
    y.append(float(_y))
x, y = np.array(x), np.array(y)
# normalization
x = (x - x.mean()) / x.std()
# plot the data
plt.figure()
plt.scatter(x, y, c="g", s=20)
plt.show()

# x0 is 100 spot between -2 and 4
x0 = np.linspace(-2, 4, 100)

test_set = (1, 3, 5)
# calculate the difference
for d in test_set:
    print(get_cost(d, x, y))

plt.scatter(x, y, c="g", s=20)
for d in test_set:
    plt.plot(x0, get_model(d, x0), label="degree = {}".format(d))
plt.xlim(-2, 4)
plt.ylim(1e5, 8e5)
# use legend method to show labels
plt.legend()
plt.show()


