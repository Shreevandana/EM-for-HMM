# Implement Expectation Maximization (EM) to train an Hidden Markov Model (HMM).

# Task


Implement EM to train an HMM for "points.dat". The observation probs should be as in EM for GMM (gaussian). Does the HMM model the data better than the original non-sequence model? What is the best number of states?

# Files


"em_hmm.py" contains the code for EM to train an HMM "README.md"

# Algorithm


EM algorithm for training an HMM is implemented. We compute forward and backward probabilities and the do the E and M steps to update the means and covariances (Baum-Welch algorithm).

Instructions for running "em_hmm.py"
To run the script "em_hmm.py" type "python3 em_hmm.py" in the commandline The default number of iterations (max_iter) is 15 and K is 2

We can also specify the maximum iterations to run using the optional argument "--max_iter", the K using "--K". The algorithm will stop if the iterations reach the maximum number of iterations or "if the increase in log-likelihood on dev is less than 1e-6".

Please note that the data file "points.dat" should be kept in the same directory as the script.

The code for obtaining the plots is present in the script.

#  Results & Interpretation

The log-likelihood was increasing as the iterations progressed as expected. The HMM models the data better than the original non-sequence model because log-likelihood is better. It was also observed that the no. of states = 2 was giving better log-likelihood on average.

Sample Output: K: 2 Reading data file: points.dat Dimensions of Train: (2, 900) Iteration 0 Iteration 5 Iteration 10 Log-Likelihoods on train: [-2988.4716034580397, -2932.9866962846577, -2927.9152023763436, -2923.1136834857239, -2919.3318370540128, -2916.8315394745259, -2915.2902909160921, -2914.2088264768331, -2913.194934378781, -2911.9814784068667, -2910.3419729507186, -2908.0124193318989, -2904.650281499029, -2899.8735400294277, -2893.4896592949281] Log-Likelihoods on dev: [-342.80544311816976, -342.06206117785291, -341.37183624357897, -340.83739550878067, -340.48123953184921, -340.24660943455626, -340.0621968932997, -339.87943924673681, -339.6709567436456, -339.41732239963386, -339.09623020249967, -338.67484902650085, -338.10528561206576, -337.33103791939652, -336.32429146753987]

# References

This was done as a homework problem in the Machine Learning class (CSC 446, Spring 2019) by Prof. Daniel Gildea (https://www.cs.rochester.edu/~gildea/) at the University of Rochester, New York.
