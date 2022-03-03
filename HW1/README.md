# Course ENPM 673- Perception for Autonomous Robots

Computer vision and perception assignments and homework submissions:

Homework 1:
Fit a line to the data using linear least square method, total least square method and RANSAC and concept of homography


## Usage
#### Problem 2:

A ball is thrown against a white background and a camera sensor is used to track its trajectory. We have a near perfect sensor tracking the ball in video1 and the second sensor is faulty and tracks the ball as shown in video2. Clearly, there is no noise added to the first video whereas there is significant noise in the second video. Assuming that the trajectory of the ball follows the equation of a parabola:

Use Standard Least Squares to fit curves to the given videos in each case. You have to plot the data and your best fit curve for each case. Submit your code along with the instructions to run it.
 
```python
git clone --recursive https://github.com/sumedhreddy90/ENPM673-Coursework.git
cd code
python3 sumedh_solution_2.py 
```

#### Problem 3:
In the above problem, we used the least squares method to fit a curve. However, if the data is scattered, this might not be the best choice for curve fitting. In this problem, you are given data for health insurance costs based on the person’s age. There are other fields as well, but you have to fit a line only for age and insurance cost data. The data is given in .csv file format and can be downloaded from here(dataset.xlsx).


Compute the covariance matrix (from scratch) and find its eigenvalues and eigenvectors. Plot the eigenvectors on the same graph as the data. Refer to this article for better understanding.


Fit a line to the data using linear least square method, total least square method and RANSAC. Plot the result for each method and explain drawbacks/advantages for each.

```python
git clone --recursive https://github.com/sumedhreddy90/ENPM673-Coursework.git
cd code
python3 sumedh_solution_3.py 
```


#### Problem 4:
Compute Homography Matrix for the below given points.

Compute SVD of the Matrix A using python.

```python
git clone --recursive https://github.com/sumedhreddy90/ENPM673-Coursework.git
cd code
python3 sumedh_solution_4.py 
```


## License
[MIT](https://choosealicense.com/licenses/mit/)
Sumedh Reddy Koppula
UID: 117386066
University of Maryland College Park
Institute of system research - MRC
