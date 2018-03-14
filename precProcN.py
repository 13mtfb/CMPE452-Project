from random import seed
from random import randrange
from random import random
from random import randint
from math import exp
from csv import reader
import csv
# Load a CSV file
# Load a CSV file
# data is in form:
#	Date,Open,High,Low,Close,Adj Close,Volume
#	Open, High, Low, Close, volume
# REMOVE Date
# REMOVE Adj Close
def loadFile(filename):
	dataset = list()
	flag = 0
	with open(filename, 'r') as file:
		CSVreader = reader(file)
		for row in CSVreader:
			if not row:
				continue

			if flag == 1:
				dataset.append(row)
			else:
				flag = 1
	return dataset

#pre Process file,
def preProc(dataset, thresh):
	data = list(dataset)

	#convert all elements to float
	for column in range(1,7):
		for row in range(0,len(dataset)):
			data[row][column] = float(dataset[row][column].strip())

	#Instead delete the time to allow for easier addition in network
	for row in range(len(data)):
		del data[row][0]
		del data[row][4]

	#Append a final column indicatiing rise fall or insinificant change of closing value
	for row in range(0,len(data)):
		diff = data[row][3]-data[row-1][3]
		if diff > thresh:
			data[row-1].append(1) #prediction class for tomorws difference
		elif diff < - thresh:
			data[row-1].append(-1)
		else:
			data[row-1].append(0)
	del data[len(data)-1]
	return data

#noramlzie dat from 0 to 1
def normalize(dataset):
	data = list(dataset)
	minList = list()
	maxList = list()
	for column in zip(*data):



		minList.append(min(column))
		#maxList.append(min(column))
		#python max() function was not working properly, finding manually
		maxi=0
		for row in column:
			if float(row) > float(maxi):
				maxi = row
		maxList.append(maxi)

	#normalize each coloumn given the list of min and max
	for row in data:
		for i in range(0,len(row)-1):
			row[i] = (float(row[i]) - float(minList[i]))/ (float(maxList[i]) - float(minList[i]))
	return data

#Add past data,
# time: number of previous entries we'd like to add
# exclu: list of columns we'd like to exclude in the repition
def addPast(dataset, time,exclu):
	#Add past data columns
	data = list(dataset)
	for row in range(time,len(data)):
		for itera in range(1,time+1):
			for col in range(0,7):
				if col not in exclu:
					data[row].append(data[row-itera][col])

	#Crop the top of the data with no added past
	cropData = list()
	for row in range(time,len(data)):
		cropData.append(data[row])

	return cropData

#Push the prediciton column  back
def pushColBack(dataset, col):
	data = list(dataset)
	for row in data:
		row.append(row[col])
		del row[col]
	return data

#write the data to csv file
def csvWrite(dataset):
	with open('readyData.csv','w') as f:
		myWrite = csv.writer(f)
		#for row in dataset:
		myWrite.writerows(dataset)

data = list()

#load a file
data = loadFile('GSPC.csv')

#Convert data to floats and convert the closing column to
data = preProc(data, 5)

#Normalize all the data
data = normalize(data)

csvWrite(data)
