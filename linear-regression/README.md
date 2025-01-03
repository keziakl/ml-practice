# Orthogonal Projection, Linear Regression, and OLS
This folder is an exploration of linear regression analysis, which is use to predict linear relationship between a scalar "reponse" (dependent variable) and one (or more) explanatory variables.


This is one of the simplest (if not the most basic) machine learning algorithms, however it should not be underestimated! The mathematical implications of linear regression's relative (Linear Projection, and equivalent: orthogonal projection) can be applied to many other, more complex algorithms! This makes linear regression a fundamental base of study before moving onto more complex machine learning algorithms and math.

# Basis
This directory's exploration is based on [this article](https://bookdown.org/ts_robinson1994/10EconometricTheorems/linear_projection.html) by Thomas Robinson (the book as a whole is moreso related to economic theory BUT is a good mathematical read rooted in basic linear algebra).

# Relationship with Linear Projection
## Linear Projection Geometrically
The intuition behind linear projection is the following:

Consider 2-dimensional vector _v_ (i.e. a finite straight line pointing in a certain direction in the _x_,_y_ plane).

Suppose there is some point _x_ which is not on this straight line but is in the same two-dimensional space. 
The projection of _x_, i.e. _Px_, is a function that returns the point on the vector _v_ that is “closest” to _x_.

If we call this point returned x̅, then we can define the Euclidean distance from x to x̅. Here is a visual:

[<img alt="orthogonal projection" width="100px" src="images/linear-regression/orthogonal-projection.png" />](https://www.google.com/)


Note: In this visual, the green line denotes the orthogonal projection (also the closest point on vector _v_ to x). The red dashed lines are non-orthogonal projections (further away in the Euclidean space than x̅).

The way that we would find the "best" _x̅_, would be to minimize the distance between it and x, meaning, we would find (out of all possible projections of _x_ onto _v_, the one which has the minimal distance.)

## Connection to Orthogonal Projection
Now, we define orthogonal projection as having the goal of minimizing the distance between the predicted values ( defined above as vector _v_) and the actual values (defined above as vector x).

We take the argmin with respect to _c_ of the euclidean distance between predicted (we define _x̅<sub>i</sub>_ to be a scalar multiplication between some optimal choice _c_ and _v<sub>i</sub>_) and the actual value _x_.

In plain English, for any point in some space, the orthogonal projection of that point onto some subspace is the point on a vector line that minimises the Euclidian distance between itself and the original point.

After differentiating with respect to c (the choice we are minimizing/optimizing), we are left with _c = (v'v)<sup>-1</sup>v'x_

We can rearrange this by remembering that the projection P<sub>x</sub>x = x̅ is _vc_. We are left with _P<sub>x</sub> = v(v'v)<sup>-1</sup>v'_

## Linear Regression
Projection matrices essentially simplify the dimensionality of some subspace by casting the points into a lower dimensional plane (ex: line to point, plane to vector). Example: Like a 2D shadow of a 3D human.

This is kind of what we try to do with linear regression. Regression is imperfect (the optimisation goal is to minimise the errors of our predictions), so in a way, we are capturing some "lower-dimensional" approximation of an outcome (y).

If we use the same minimizing and differentiating to find the optimal prediction "y" based on the true "y", then using the definition of a linear function _y = XB + e_ (where B is the vector of coefficients we want to estimate, e is the difference between the predicted value and the observed, true value), we actually get something really similar to what we saw with orthogonal projection:

_y^ = X(X'X)<sup>-1</sup>X'y_

Therefore, if we define the projection P<sub>y</sub> to equal _X(X'X)<sup>-1</sup>X'_, then the _Py_ is equal to _y^_!

# Geometric Analysis (i think it's pretty cool)
Let's picture a bivariate regression problem with 3 observations.

_Y = (2, 3, 2)_
_X = (3, 1, 1)_
_c = (1, 1, 1)_

Usually, we would use each variable (Y, X) as dimensions, and each data point would be an observation (ex: (x = 2, y =3)), and we would ignore the constants column since it's always the same.

However, another way to visualize this is to treat each observation (row in the table, obs<sub>1</sub>, obs<sub>2</sub>, obs<sub>3</sub>) as dimension and each variable _y = (2, 3, 2)_ as a vector. This allows us to define the column space of X and c (_col(X, c)_), which is a linear combination of vectors X and c. Because X and c are linearly independent and there is only 2 of them (in a 3D space), the resulting _span_ is a 2D plane in the 3D space (the 3 observations we have).

[<img alt="orthogonal projection" width="100px" src="images/linear-regression/3d-plane.png" />](https://www.google.com/)

Vector _y_ lies of the plane, but we want to approximate it using the _span_ (think: in terms of) X and c. Therefore, we find the orthogonal projection (the "best" vector that lies on the plane). We know that the plane is composed of linear combinations of _B<sub>1</sub>X + B<sub>0</sub>c_. The scalar coefficents (the Bs) are the regression coefficients derived by Ordinary Least Squares! Therefore, geometrically, orthogonal projection is equivalent to linear regression.