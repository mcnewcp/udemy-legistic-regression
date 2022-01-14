# What does the output of logistic regression actually mean?

The output of logistic regression is always between 0 and 1 and represents the probability that y = class 1 given x
p(y=1 | x)

and so the probabilities sum up to 1
p(y=1 | x) + p(y=0 | x) = 1

and so rounding the output, using a 0.5 threshold, results in either 0 or 1 and giving the prediction directly.