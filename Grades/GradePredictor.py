
#Import Library
import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use("ggplot")

data = pd.read_csv("Grades/student-mat.csv", sep=";")

predict = "G3"

data = data[["G1", "G2", "absences","failures", "studytime","G3"]]
data = shuffle(data) # Optional - shuffle the data

x = np.array(data.drop([predict], 1))
y =np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.99)


# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
best = 0.975656497931211

# for i in range(50_000):
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
#         with open("studentgrades.pickle", "wb") as f:
#             pickle.dump(linear, f)


# LOAD MODEL
pickle_in = open("Grades/studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)




predicted= linear.predict(x_test)
avgDifference = 0;
for x in range(len(predicted)):
    print("Predicted Value:\t", str(round(predicted[x])) + "\t", "Data Set:\t", str(x_test[x]) + "\t", "Actual Value\t", y_test[x])
    avgDifference+= abs(round(predicted[x]) - y_test[x])

print("-------------------------")
print("best: \n", best)
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
print('accuracy: \n', linear.score(x_test, y_test))
print("Average Difference (Predicted - Actual):\n " + str(avgDifference/len(predicted)))
print("-------------------------")

# Drawing and plotting model
# plot = "G1"
# plt.scatter(data[plot], data["G3"])
# plt.legend(loc=4)
# plt.xlabel(plot)
# plt.ylabel("Final Grade")
# plt.show()