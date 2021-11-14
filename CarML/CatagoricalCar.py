import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("CarPrice_Assignment.csv", sep=",")
print(data.head())

pl = preprocessing.LabelEncoder();
#Type Data
fueltype = pl.fit_transform(list(data["fueltype"]))
aspiration = pl.fit_transform(list(data["aspiration"]))
doornumber = pl.fit_transform(list(data["doornumber"]))
carbody = pl.fit_transform(list(data["carbody"]))
drivewheel = pl.fit_transform(list(data["drivewheel"]))
enginelocation = pl.fit_transform(list(data["enginelocation"]))
enginetype = pl.fit_transform(list(data["enginetype"]))
cylindernumber = pl.fit_transform(list(data["cylindernumber"]))
fuelsystem = pl.fit_transform(list(data["fuelsystem"]))
enginetype = pl.fit_transform(list(data["enginetype"]))

#Number Data
# price = pl.fi(list(data["price"]))
# wheelbase = (list(data["wheelbase"]))
# carlength = (list(data["carlength"]))
# carwidth = (list(data["carwidth"]))
# carheight = (list(data["carheight"]))
# curbweight = (list(data["curbweight"]))
# enginesize =  (list(data["enginesize"]))
# horsepower =  (list(data["horsepower"]))
# peakrpm =  (list(data["peakrpm"]))
# citympg = (list(data["peakrpm"]))
# highwaympg = (list(data["peakrpm"]))


X = list(zip(fueltype,aspiration,doornumber,carbody,drivewheel,enginelocation,enginetype,cylindernumber,fuelsystem,enginetype))
Y = carbody

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
boodies = ["convertible","hatchback", "sedan","wagon","hardtop"]

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
pridicted = model.predict(x_test);

for x in range(len(pridicted)):
    print("Predicated Body Type:\t" + str(boodies[pridicted[x]]) + "\t", "Data\t" + str(x_test[x]) + "\t", "Actual: " + str(boodies[y_test[x]]))
print("----------")
print(acc)
print("----------")

