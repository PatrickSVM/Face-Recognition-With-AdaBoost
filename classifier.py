import numpy as np 
from functools import partial
from time import sleep
import sys



def WeakClassifier(T, P, X):
    """ WeakClassifier
    
    Classify images using a decision stump.
    
    Takes a vector X of scalars obtained by applying one Haar-feature to all
    training images. Classifies the examples using a decision stump with
    cut-off T and polarity P. Returns a vector C of classifications for all
    examples in X.

    Inputs:
            T - Threshold cutoff (scalar)
            P - Polarity, either 1 or -1 (scalar)
            X - Feature vector for one Haar-feature and all Training examples (vector)

    Output:
            C - Predicted labels for each sample (vector)
    """
    

    C = np.ones(X.shape[0])
    idx = np.argwhere((P*X) < (P*T)) # If smaller than threshold, assign class -1
    C[idx] = -1
    
    return C




def WeakClassifierError(C, D, Y):
    """ WeakClassifierError
    
    Calculate the error of a single decision stump.
    
    Takes a vector C of classifications from a weak classifier, a vector D
    with weights for each example, and a vector Y with desired
    classifications. Calculates the weighted error of C, using the 0-1 cost
    function.

    Inputs:
            C - Predicted labels for each sample (vector)
            D - Weights for each example (vector)
            Y - Desired output, real labels (vector)

    Output:
            E - Weighted error (scalar)
    """

    E = np.dot(D, (C != Y)) # Sum (1 * weight_i) for all positions where C_i is different from Y_i

    return E





def TrainWeakClassifier(X, D, Y):
    """ TrainWeakClassifier
    
    Trains a weak classifier from the given feature vector X by choosing the best split threshold and polarity, that minimizes the 
    decision stump's weighted error.
    
    Inputs:
            X - Feature vector for one Haar-feature and all Training examples (vector)
            D - Weights for each example (vector)
            Y - Desired output, real labels (vector)

    Output:
            best_classifier - Best decision stump for feature X given D and Y (function wit T=T_opt, P=P_opt)
    """
    
    thresh_poss = np.unique(X) # Get all possible thresholds from X
    
    new_classifier = partial(WeakClassifier, P=1, X=X) # function that trains a WeakClassifier on X with P=1
    
    # Get a list of the weighted errors of all possible decision stumps using all possible thresholds
    result = np.array(list(map(lambda labels: WeakClassifierError(labels, D=D, Y=Y), map(new_classifier, thresh_poss))))
    
    # Change polarity where the weighted error is > 0.5:
    polarities = np.ones(X.shape[0])
    idx_wrong_polar = np.argwhere(result > 0.5) 
    result[idx_wrong_polar] = 1 - result[idx_wrong_polar]
    polarities[idx_wrong_polar] = -1
    
    # Find threshold that minimizes weighted error
    best_thresh_idx = np.argmin(result)
    best_thresh = thresh_poss[best_thresh_idx]
    polarity = polarities[best_thresh_idx]
    
    # Train the final best decision stump
    best_classifier = partial(WeakClassifier, T=best_thresh, P=polarity)
    
    return best_classifier





def AdaBoost(X, Y, n_classifiers, random_choice=False):
    """ AdaBoost
    
    Implementation of the AdaBoost algorithm. 
    
    Performs AdaBoost algorithm and returns the final StrongClassifier, the associated feature matrix,
    the indexes of the chosen Haar-Features and the referring alpha values. 
    
    Inputs:
            X             - Feature matrix for Haar-feature-values and all Training examples 
                            (matrix: num_Haar_features x num_samples)
            Y             - Desired output, real labels (vector)
            n_classifiers - Number of decision stumps to be trained (scalar)
            random_choice - Whether to use classical AdaBoost approach or random feature selection in iterations (boolean)

    Output:
            StrongClass - Final strong classifier composed of the trained decision stumps (function)
    """
    
    
    def StrongClassifier(X, weak_classifiers, alphas, n_class):
        """ StrongClassifier
        
        Return the classification result of a strong classifier composed
        of the best n_class weak classifiers in weak_classifiers, so the ones
        contributing the most to the final result (importance measured by the
        respective alpha-values). 
        """
        
        assert n_class <= len(weak_classifiers), "Number of classifiers must be <= number of total WeakClassifiers."
        
        results = np.zeros(shape=(X.shape[1], n_class))  # Result matrix, Row_i = Example_i, Col_j = Result of Classifier_j
        
        classif_idx = np.argsort(-alphas)[:n_class] # Get index of n_class most important WeakClassifiers, 
                                                    # so index of n_class highest alpha values 

        for i, classifier in enumerate(weak_classifiers[classif_idx]):
            res = classifier(X=X[classif_idx[i], ]) # Compute result from classifier with index classif_idx[i]
            results[:, i] = res
        
        # Compute final decision for all examples as the sign of the alpha-weighted sum over all n_class decision stumps
        classes = np.sign(np.sum(results*alphas[classif_idx], axis=1)) 
        
        return classes
    

    n = X.shape[1]               # Number of training samples
    num_features = X.shape[0]    # Number of features 
    D = np.full(n, 1/n)          # Get first weight matrix with uniform weights

    classifiers = []  # Store all classifiers
    alphas = []       # Store alpha values
    chosen_idx = []   # Store index of the feature chosen, to remember which feature array needs to be used for which classifier
    
    
    if not random_choice: # Progress bar 
        print("AdaBoost Training Progress\n")
        sys.stdout.write('\r')
        sys.stdout.write("[%-40s] %d%%" % ('='*(0), 2.5*(0)))
        sys.stdout.flush()
    
    
    for n in range(n_classifiers): # n_classifiers iterations of AdaBoost
        
        if random_choice:  # Adjusted AdaBoost Verison: Choose a filter at random
            min_err_idx = np.random.randint(0, max(1, num_features-1))
        
        else: # Classical AdaBoost Version: Choose the filter that minimizes the current weighted error 
            all_errors = np.apply_along_axis(lambda row: WeakClassifierError(TrainWeakClassifier(X=row, D=D, Y=Y)(X=row), D, Y), 
                                        axis=1, 
                                        arr=X)  # Get error for the best classifier for all possible features
            min_err_idx = np.argmin(all_errors) # Choose the feature with the minimal weighted error
            
            # Print progress
            prog = int((((n+1)*100)/n_classifiers) // 2.5)
            
            sys.stdout.write('\r')
            sys.stdout.write("[%-40s] %.1f%%" % ('='*(prog), 2.5*(prog)))
            sys.stdout.flush()

        
        chosen_idx.append(min_err_idx)   
        best_class = TrainWeakClassifier(X[min_err_idx, :], D, Y)   # Train the chosen classifier
        Y_pred = best_class(X=X[min_err_idx])   # Get prediciton 

        error = WeakClassifierError(Y_pred, D, Y)   # Get error of the current prediction 

        alpha = 0.5 * np.log((1-error)/error)               # Update alpha value
        D = D * (np.exp(-alpha * np.multiply(Y, Y_pred)))   # Update weights
        D = D/sum(D)                                        # Normalize weights

        classifiers.append(best_class)
        alphas.append(alpha)

    feature_matrix = X[np.array(chosen_idx), ]   # Get feature matrix, first row contains feature-vector for first chosen feature,
                                                 # second row contains feature-vector for second chosen feature, ...
                                                 # If same feature is chosen twice, rows can double in the matrix
        
    alphas, classifiers = np.array(alphas), np.array(classifiers)
    StrongClass = partial(StrongClassifier, weak_classifiers=classifiers, alphas=alphas, n_class=len(classifiers)) 
    
    return StrongClass, feature_matrix, chosen_idx, alphas