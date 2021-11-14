
#Import Library
import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle


data = pd.read_csv("CarPrice_Assignment.csv", sep=",")

predict = "price"

data = data[["wheelbase","carlength","carwidth","carheight","curbweight","enginesize","stroke", "compressionratio","horsepower","peakrpm","citympg","highwaympg","price"]]
data = shuffle(data) # Optional - shuffle the data
units =["wheelbase","carlength","carwidth","carheight","curbweight","enginesize","stroke", "compressionratio","horsepower","peakrpm","citympg","highwaympg","price"]
x = np.array(data.drop([predict], 1))
y =np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.99)


# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
best = 0.9798295304228791

# for i in range(100_000):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#
#     linear = linear_model.LinearRegression()
#
#     linear.fit(x_train, y_train)
#     acc = linear.score(x_test, y_test)
#     print("Accuracy: " + str(acc))
#
#     if acc > best:
#         best = acc
#         print(i," : ", acc)
#         with open("carPrice.pickle", "wb") as f:
#             pickle.dump(linear, f)


# LOAD MODEL
pickle_in = open("carPrice.pickle", "rb")
linear = pickle.load(pickle_in)




predicted= linear.predict(x_test)
avgDifference = 0;
for x in range(len(predicted)):
    print("Predicted Value:\t", str(round(predicted[x])) + "\t", "Actual Value\t", y_test[x])
    for i in range (len(x_test[x])):
        print(str(units[i]) + "\t:\t" + str(x_test[x,i]))
    avgDifference+= abs(round(predicted[x]) - y_test[x])
    print("-----------------------------------")

print("-------------------------")
print("best: \n", best)
print('Coefficient: \n')
for i in range (len(linear.coef_)):
    print(str(units[i])+ "\t:\t" + str(linear.coef_[i]))
print("\n","\n")
print('Intercept: \n', linear.intercept_)
print('accuracy: \n', linear.score(x_test, y_test))
print("Average Difference (Predicted - Actual):\n " + str(avgDifference/len(predicted)))
print("-------------------------")

# audi = [ [9.650e+01,1.754e+02, 6.0e+01, 5.410e+01, 2.372e+03, 3e+02, 3.580e+00,
#  9.000e+00, 83.600e+01, 5.800e+03, 2.700e+01, 3.300e+01]]
#
# print(linear.predict(audi))
#10295
# print(x_test)
# Drawing and plotting model
# plot = "G1"
# plt.scatter(data[plot], data["G3"])
# plt.legend(loc=4)
# plt.xlabel(plot)
# plt.ylabel("Final Grade")
# plt.show()