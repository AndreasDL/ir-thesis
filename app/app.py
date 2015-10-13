import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

trainPersons = 1 # one set = one person
x_train = np.zeros((40, 1)) #40 x 1 feature
y_train = np.zeros((40, 1)) #40 x 1 label

#load trainSet
for i in range(trainPersons):
	fname = 'dataset/s'
	if i+1 < 10:
		fname += '0' 
	fname += str(i+1) + '.dat'
	
	with open(fname,'rb') as f:
		p = pickle._Unpickler(f)
		p.encoding= ('latin1')
		data = p.load()
		#structure
		#data['labels'][video , attribute]
		#data['data'][video, channel, value]

		y_train = data['labels'][:,1] #only valence needed
		#structure y_train[video]

		#get features
		#different left/right right more active than left
		#calculate sum_left & sum_right & sum_total = sum_left + sum_right
		#use feature= sum_right / sum_total
		sum_right = np.zeros(40) #40 videos
		sum_left = np.zeros(40)
		for j in range(len(data['data'])): #for each video
			sum_right[j] += np.sum(data['data'][j,16,:])
			sum_right[j] += np.sum(data['data'][j,19,:])
			sum_right[j] += np.sum(data['data'][j,21,:])
			sum_right[j] += np.sum(data['data'][j,24,:])
			sum_right[j] += np.sum(data['data'][j,26,:])
			sum_right[j] += np.sum(data['data'][j,28,:])
			sum_right[j] += np.sum(data['data'][j,30,:])
			sum_right[j] += np.sum(data['data'][j,31,:])
			
			sum_left[j] += np.sum(data['data'][j,1,:])
			sum_left[j] += np.sum(data['data'][j,3,:])
			sum_left[j] += np.sum(data['data'][j,4,:])
			sum_left[j] += np.sum(data['data'][j,7,:])
			sum_left[j] += np.sum(data['data'][j,8,:])
			sum_left[j] += np.sum(data['data'][j,11,:])
			sum_left[j] += np.sum(data['data'][j,14,:])

#			print(sum_right)
#			print(sum_left)
#			print((sum_right/(sum_right + sum_left)))
			#fraction of activity at right side
			x_train[j,0] = sum_right[j]/(sum_right[j] + sum_left[j] + 1)

#testset
testPersons = 1 # one set = one person
x_test = np.zeros((40, 1)) #40 x 1 feature
y_test = np.zeros((40, 1)) #40 x 1 label

#load testSet
for i in range(testPersons):
	fname = 'dataset/s'
	if i+1+trainPersons < 10:
		fname += '0' 
	fname += str(i+1+trainPersons) + '.dat'
	
	with open(fname,'rb') as f:
		p = pickle._Unpickler(f)
		p.encoding= ('latin1')
		data = p.load()
		#structure
		#data['labels'][video , attribute]
		#data['data'][video, channel, value]

		y_test = data['labels'][:,1] #only valence needed
		#structure y_train[video]

		#get features
		#different left/right right more active than left
		#calculate sum_left & sum_right & sum_total = sum_left + sum_right
		#use feature= sum_right / sum_total
		#don't use sum(x['data']), cuz we ignore center electrodes
		sum_right = np.zeros(40)
		sum_left = np.zeros(40)
		for j in range(len(data['data'])): #for each video
			sum_right[j] += np.sum(data['data'][j,16,:])
			sum_right[j] += np.sum(data['data'][j,19,:])
			sum_right[j] += np.sum(data['data'][j,21,:])
			sum_right[j] += np.sum(data['data'][j,24,:])
			sum_right[j] += np.sum(data['data'][j,26,:])
			sum_right[j] += np.sum(data['data'][j,28,:])
			sum_right[j] += np.sum(data['data'][j,30,:])
			sum_right[j] += np.sum(data['data'][j,31,:])
			
			sum_left[j] += np.sum(data['data'][j,1,:])
			sum_left[j] += np.sum(data['data'][j,3,:])
			sum_left[j] += np.sum(data['data'][j,4,:])
			sum_left[j] += np.sum(data['data'][j,7,:])
			sum_left[j] += np.sum(data['data'][j,8,:])
			sum_left[j] += np.sum(data['data'][j,11,:])
			sum_left[j] += np.sum(data['data'][j,14,:])

			#fraction of activity at right side
			x_test[j,0] = sum_right[j]/( sum_right[j] + sum_left[j] + 1 )


#starting from here we have
#a trainset consisting of trainPersons 
#y_train: holds all the valence values for each movie
#x_train: holds all the right fraction values for each movie
#
#a testset consisting of testPersons
#y_train: holds all valence values for each movie
#x_train: holds all the right fraction values for each movie

#starting here we will
#perform linear regression
# Create linear regression object
regr = linear_model.LinearRegression()

print(len(x_train))
print(len(y_train))
# Train the model using the training sets
regr.fit(x_train, y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)

# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(x_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x_test, y_test))

# Plot outputs
plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, regr.predict(x_test), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()