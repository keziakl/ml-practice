"""
Explores the relationship between linear regression (OLS) and orthogonal projection
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

# Let's make up a simple dataset to explore this, based on section 3.3.1 of https://bookdown.org/ts_robinson1994/10EconometricTheorems/linear_projection.html
X = np.array([[3, 1], [1, 1], [1, 1]]) # 3 observations, 2 features each
y = np.array([1, 2, 2]) # 3 observations, scalar response

# Perform the linear regression based on the least squares formula
# Note: Beta Hat is equal to B0 = slope, B1 = intercept (since X1 is a constant)
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
y_hat = X @ beta_hat

# Orthogonal Projection
# We define the projection matrix to be: P = X(X'X)^(-1)X'
P = X @ np.linalg.inv(X.T @ X) @ X.T

# We compute y_hat by using the projection matrix
y_hat_proj = P @ y

# The residual (squared is what we want to minimize)
residual = y - y_hat_proj

# We verify that the X' * residual is close to orthogonal to the column space of X, meaning, it is 0
orthogonality = np.allclose(X.T @ residual, 0)
print(f"Residuals are orthogonal to the column space of X: {orthogonality}")

# Now, let's visualize each of these vectors
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Let's visualize y (the observed values), y_hat (the predicted values), and the 2D plane (span of X)
ax.scatter(2, 3, 2, label='y', color='red')
# Note: only one of y_hat or y_hat_proj is needed, but we visualize to show they are the same
ax.quiver(0, 0, 0, y_hat[0], y_hat[1], y_hat[2], color='red', label='y_hat', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, y_hat_proj[0], y_hat_proj[1], y_hat_proj[2], color='blue', label='y_hat_proj', arrow_length_ratio=0.1)

# Style stuff
ax.pbaspect = 'orthographic'
ax.set_xticks([-1, 0, 1, 2, 3])
ax.set_yticks([-2, 0, 1, 2, 3])
ax.set_zticks([-1, 0, 1, 2, 3])
# Each axis is one observation
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')

ax.legend()
plt.show()

# Now, let's check that our slope and intercept (Beta Hat) are the same as the ones we would get from scipy's linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(X[:, 0], y)
beta_hat_scipy = np.array([slope, intercept])
print(f"Assert that beta_hat and beta_hat_scipy are the same: {np.allclose(beta_hat, beta_hat_scipy)}")