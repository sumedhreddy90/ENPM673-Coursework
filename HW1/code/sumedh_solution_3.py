import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import pandas as pd
from numpy import linalg as linear

# Importing data from csv file using pandas

data = pd.read_csv('sumedh.csv')

# storing data in indivdual lists
ages = data['age'].to_numpy(int)
insurance_cost = data['charges'].to_numpy(np.float64)

# Creating Matrix with age and cost dataw
data = np.vstack((ages, insurance_cost)).T

# Formulating Covariance 
def covariance(x, y):
    x_i, y_i = x.mean(), y.mean()
    return np.sum((x - x_i)*(y - y_i))/(len(x) - 1)

# Covariance matrix
def covariance_mat(matrix):
    return np.array([[covariance(matrix[0], matrix[0]), covariance(matrix[0], matrix[1])], 
                     [covariance(matrix[1], matrix[0]), covariance(matrix[1], matrix[1])]])

def standardLS(points):
   
    x_axis=points[:,0]
    y_axis=points[:,1]

    # Implementing line equation
    M = np.stack((x_axis, np.ones((len(x_axis)), dtype = int)), axis = 1) 
    
    M_transpose = M.transpose()
    MTM = M_transpose.dot(M)
    MTY = M_transpose.dot(M)
    ls_estimate = (np.linalg.inv(MTM)).dot(MTY)
    ls_value= M.dot(ls_estimate)
    
    return ls_value

def leastSquares(E,Y):

    E_transpose = E.transpose()
    ETE = E_transpose.dot(E)
    ETY = E_transpose.dot(Y)
    ls_est = (np.linalg.inv(ETE)).dot(ETY)
    return ls_est

def totalLeastSquares (x_ls,y_ls):

    #Constructing U matrix 
    n = len(x_ls)
    
    x_mean=np.mean(x_ls)
    y_mean=np.mean(y_ls)
    
    U=np.vstack(((x_ls-x_mean),(y_ls-y_mean))).T

    # Evaluating UtU
    UTU=np.dot(U.transpose(),U)
    
    # Calculating coefficents of line equation y=ax+b
    beta=np.dot(UTU.transpose(),UTU)
    
    # Calculating eigen values of beta
    val,vect=linear.eig(beta)
    
    # Sorting the index of smallest eigen value
    ind =np.argmin(val)

    # eigen vector for smallest egien value
    coef =vect[:,ind]
    
    a,b= coef
    Ym =a*x_mean+b
    
    total_least_square=[]
    for i in range(0,n):
        temp=(Ym-(a*x_ls[i]))/b
        total_least_square.append(temp)
    
    return total_least_square

def ransacGenerator(ransac_array):

    x=ransac_array[:,0]
    y=ransac_array[:,1]
    
    # generating a random line array
    a =np.stack((x,np.ones((len(x)),dtype=int)),axis=1)
    
    #Setting threshold for inliers and outliers
    threshold=np.std(y)/2
    
    #Fiting a  line to the selected 2 points
    ransac_voting =ransacVoter(a,y,2, threshold)
    
    ransac_=a.dot(ransac_voting)

    return ransac_

def ransacVoter(line_array, y, size, threshold):
    max_i=math.inf
    iteration=0
    desired_prob=0.95 
    best_fit=None
    max=0 
    outlier=0

    ransac_data=np.column_stack((line_array,y))
    dsize=len(ransac_data)

    # finding points close to line
    while max_i >iteration:
        
        np.random.shuffle(ransac_data)
        data_set=ransac_data[:size,:]
        
        #creating line using least square
        x_t=data_set[:,:-1]
        y_t=data_set[:,-1:]
        voting_scheme=leastSquares(x_t,y_t)

        #counting inliers with defined range of threshold
        inliers=line_array.dot(voting_scheme)
        error =np.abs(y-inliers.T)
        count=np.count_nonzero(error<threshold)
        print("number of Inliers: ", count, "for iteration ", iteration)
        
        #best fit
        if count >max:
            max=count
            best_fit=voting_scheme
            
        #Outliers
        outlier=1-count/dsize
        max_i =math.log(1-desired_prob)/math.log(1-(1-outlier)**size)
        
        iteration+=1
        
    return best_fit    

#Covariance for given data
cov =covariance_mat(data.T) 

c_value, c_vector = np.linalg.eig(cov)

# Plotting data scatter plot
plt.scatter(data[:, 0], data[:, 1])


origin=[np.mean(data[:,0]),np.mean(data[:,1])]
eigen_1=c_vector[:,0]
eigen_2=c_vector[:,1]

#Calculating Least Square 
ls = standardLS(data)
#Calculating total Least Square
total_ls = totalLeastSquares(ages, insurance_cost)
#Calculating RANSAC
rans = ransacGenerator(data)

#Plotting data and curve fitting
#Plotting Covariance vector and egien vectors
plt.figure(1)
plt.quiver(*origin, *eigen_1, color=['r'], scale=15)
plt.quiver(*origin, *eigen_2, color=['b'], scale=15)
plt.title('Covariance Matrix')
plt.xlabel("age")
plt.ylabel('Insurance cost')

#Curve fitting using Least Square 
plt.figure(2)
plt.xlabel('Age')
plt.ylabel('Insurance cost')
plt.title('curve fitting using Least Square ')
plt.scatter(ages,insurance_cost,c='g',label='given data')
plt.plot(ages,ls, c='red', label='Linear Least Squares')
plt.legend()

#Curve fitting using Total Least Square 
plt.figure(3)
plt.xlabel('Age')
plt.ylabel('Insurance cost')
plt.title('curve fitting using Total Least Square ')
plt.scatter(ages,insurance_cost,c='y',label='given data')
plt.plot(ages,total_ls, c='red', label='Linear Least Squares')
plt.legend()

plt.figure(4)
plt.xlabel('Age')
plt.ylabel('Insurance cost')
plt.title('curve fitting using RANSAC ')
plt.scatter(ages,insurance_cost,c='y',label='given data')
plt.plot(ages,rans, c='red', label='RANSAC')
plt.legend()

plt.show()