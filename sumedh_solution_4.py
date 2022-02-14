from this import d
import numpy as np
from numpy import diagonal, linalg as linear
import matplotlib.pyplot as plt


x1,x2,x3,x4,y1,y2,y3,y4= 5,150,150,5,5,5,150,150
xp1,xp2,xp3,xp4,yp1,yp2,yp3,yp4 = 100,200,220,100,100,80,80,200

A = np.array([[-x1,-y1,-1,0,0,0,x1*xp1,y1*xp1,xp1],
              [0,0,0,-x1,-y1,-1,x1*yp1,y1*yp1,yp1],
              [-x2,-y2,-1,0,0,0,x2*xp2,y2*xp2,xp2],
              [0,0,0,-x2,-y2,-1,x2*yp2,y2*yp2,yp2],
              [-x3,-y3,-1,0,0,0,x3*xp3,y3*xp3,xp3],
              [0,0,0,-x3,-y3,-1,x3*yp3,y3*yp3,yp3],
              [-x4,-y4,-1,0,0,0,x4*xp4,y4*xp4,xp4],
              [0,0,0,-x4,-y4,-1,x4*yp4,y4*yp4,yp4]])

def svdCalculator(A):

    # calculating U matrix
    A_transp = A.transpose()
    AA_transp = A.dot(A_transp)
    
    # computing eigen values and eigen vectors
    eig_values , eig_vector = linear.eig(AA_transp)
    
    #sorting eigen values and vectors
    eig_sort = eig_values.argsort()[::-1]
    u_sorted = eig_values[eig_sort]
    U_Matrix = eig_vector[:,eig_sort]
        
    # computing V transpose matrix

    A_transp_A = A_transp.dot(A)
    Veig_val,Veig_vect = linear.eig(A_transp_A)
    Vsort = Veig_val.argsort()[::-1]
    V_Matrix = Veig_vect[:,Vsort]
    V_trasp = V_Matrix.transpose()

    # flitering non positive eigen values
    for i in range(len(u_sorted)):
        if u_sorted[i]<=0:
            u_sorted[i]*=-1
            
    #Computing Sigma Matrix

    diagonal = np.diag((np.sqrt(u_sorted)))
    sigma_matrix = np.zeros_like(A).astype(np.float64) 
   
    for i in range(len(sigma_matrix)):
        sigma_matrix[i][i]=diagonal[i][i]
    
    #computing homography
    homography = V_Matrix[:,8]  
    homography = np.reshape(homography,(3,3))

    return V_trasp, U_Matrix, sigma_matrix, homography

V_star, U, sig, homo = svdCalculator(A)

print("Homography Matrix: ", homo)






