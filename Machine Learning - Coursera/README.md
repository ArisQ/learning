# Notes for Machine Learning on coursera

[course website](https://www.coursera.org/learn/machine-learning/)

* Definition of machine learning

    *  "the field of study that gives computers the ability to learn without being explicitly programmed."  -  Arthur Samuel

    * "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E." - Tom Mitchell

* Types of machine learning

    * Supervised learning

        * Regression problems

        * Classification problems

    * Unsupervised learning

        * Clustering

        * Non-clustering

            * Cocktail party problem

* Model Representation

    * notation

        * $m$ : Number of traing examples

        * $x$'s : input variable/features

        * $y$'s : output variable/ target variable

        * $x^{(i)}$ : $i^{th}$ input

        * $y^{(i)}$ : $i^{th}$ output

        * $(x^{(i)},y^{(i)})$ : $i^{th}$ training example

    * model struct

        ![x](images/model.png)

        $h$ stand for hypothesis, for linear regression $h_\theta(x)=\theta_0+\theta_1 x$.

* Cost Function

    We can measure the accuracy of our hypothesis function by using a **cost function**.

    $$\min_{\theta_0,\theta_1}\!\mathrm{imize}J(\theta_0,\theta_1)$$
    <!--- $$\overset{\theta_0,\theta_1}{minimize}J(\theta_0,\theta_1)$$ --->
    $$J(\theta_1,\theta_2)=\frac{1}{2m}\sum_{i=1}^n(h_\theta(x^{(i)})-y^{(i)})^2$$
    Also called Squared error function/Mean squared error:

* Parameter learning

    * Gradient descent algorithm

        repeate until convergence {
            $\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1)$ (for $j=0$ and $j=1$)
        }

        * $\alpha$ : learning rate

            * if $\alpha$ is too small, gradient descent can be slow.

            * if $\alpha$ is too large, gradient descent can overshoot the minimum. It may fail to converge, or even diverge.

        * simultaneously  update(right implementation)

        * non-simultaneous update (wrong implementation,another algorithm)

        * Batch gradient descent
