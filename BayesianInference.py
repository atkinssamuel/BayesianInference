import numpy as np
import math 
import scipy
from matplotlib import pyplot as plt
from data_utils import load_dataset
from IPython import display
import random
import copy
from numpy.random import multivariate_normal as multivariateNormal
from scipy.stats import multivariate_normal
inf = 10000000

sigmoid = lambda z: np.divide(1, np.add(1, np.exp(-z)))
logG = lambda H: 1/2*np.log(np.linalg.det(-1*H))-len(H)/2*np.log(2*np.pi)
priorLog = lambda w, sigma: -len(w)/2*np.log(2*np.pi)-len(w)/2*np.log(sigma)-1/(2*sigma)*np.dot(w.T, w)
priorGradient = lambda w, sigma: -1/sigma*w
priorGradient2 = lambda w, sigma: -1/sigma*np.eye(len(w))
priorTerm = lambda w, variance: np.product(1.0/np.sqrt(2.0*np.pi*variance)*np.exp(-np.square(w)/(2*variance)))
logLikelihood = lambda result, y: np.dot(y.T, np.log(sigmoid(result))) + np.dot(np.subtract(1, y).T, np.log(np.subtract(1, sigmoid(result))))
dataLikelihood = lambda w, x, y: np.product((1.-np.expand_dims(1. / (1. + np.exp(-x.dot(w))), axis = 1))**(~y) * np.expand_dims(1. / (1. + np.exp(-x.dot(w))), axis = 1)**y, axis=0)
eachLikelihood = lambda w, x, y: (1.-np.expand_dims(1. / (1. + np.exp(-x.dot(w))), axis = 1))**(1 - y)*np.expand_dims(1./(1. + np.exp(-x.dot(w))), axis = 1)**y
proposedPDF = lambda w, s: np.expand_dims(np.random.multivariate_normal(mean = np.squeeze(w), cov = np.eye(np.shape(np.squeeze(w))[0])*s), axis =1)

def createMatrix(x):
    x_matrix = np.ones((len(x), len(x[0])+1))
    x_matrix[:, 1:] = x
    return x_matrix

def likelihoodGradient(x, result, y):
    gradient = np.zeros(np.shape(x[0]))
    for i in range(len(result)):
        gradient += (y[i] - sigmoid(result[i])) * x[i]
    return gradient

def likelihoodGradient2(x, result):
    s = np.multiply(sigmoid(result), sigmoid(result) - 1)
    H = np.zeros((len(x[0]), len(x[0])))
    for j in range(len(result)):
        H = np.add(H, s[j] * np.outer(x[j], x[j].T))
    return H

def generateFolds(data):
    size = len(data)
    foldSet = copy.deepcopy(list(data))

    f1 = [foldSet[math.ceil(size/5):], foldSet[:math.ceil(size/5)]]
    
    t2 = foldSet[:math.ceil(size/5)]
    t2.extend(foldSet[math.ceil(size/5 * 2):])
    f2 = [t2, foldSet[math.ceil(size/5):math.ceil(size/5*2)]]                                           
                                         
    t3 = foldSet[:math.ceil(size/5*2)]
    t3.extend(foldSet[math.ceil(size/5*3):])
    f3 = [t3, foldSet[math.ceil(size/5*2):math.ceil(size/5*3)]]                                             
    
    t4 = foldSet[:math.ceil(size/5*3)]
    t4.extend(foldSet[math.ceil(size/5*4):])
    f4 = [t4, foldSet[math.ceil(size/5*3):math.ceil(size/5*4)]]                                            
    
    f5 = [foldSet[:math.ceil(size/5*4)], foldSet[math.ceil(size/5*4):]]
    
    return [f1, f2, f3, f4, f5]

def crossValidation(x_train, y_train, meanList, H, variance):
    # Initializing optimal weight and accuracy:
    optimalAccuracy = -inf
    optimalW = None

    # Shuffling:
    x_y_train = np.hstack((x_train, y_train))
    np.random.shuffle(x_y_train)
    x_train = x_y_train[:,:5]
    y_train = x_y_train[:,5]
    y_train = np.expand_dims(np.array(y_train, dtype = bool), axis = 1)
    x_train = list(x_train)
    y_train = list(y_train)

    # Initializing folds for 5-fold cross-validation:
    x_folds = generateFolds(x_train)
    y_folds = generateFolds(y_train)
    print("Cross Validation:")
    for mean in meanList:
        accuracyList = []
        wProposed = multivariateNormal(mean=mean, cov=np.linalg.inv(-H), size=1000)
        qDistribution = np.reshape(multivariate_normal.pdf(wProposed, mean=mean, cov= np.linalg.inv(-H)), (1000, 1))
        print("\nMean Vector = {}".format(mean))
        for j in range(5):
            print("Fold = {}".format(j + 1))
            correctCount = 0

            # Creating the fold:
            x_fold, y_fold = x_folds[j], y_folds[j]
            x_cross_train, x_cross_test = np.array(x_fold[0]), np.array(x_fold[1])
            y_cross_train, y_cross_test = np.array(y_fold[0]), np.array(y_fold[1])

            # Predicting:
            posteriorLPrediction = posteriorPredictor(x_cross_train, y_cross_train, x_cross_test, y_cross_test, wProposed, qDistribution, variance)
            
            # Looping through each prediction:
            for i in range(len(posteriorLPrediction)):
                posteriorPrediction = np.squeeze(posteriorLPrediction[i])
                y = np.squeeze(y_cross_test[i]) 
                prediction = y
                if(posteriorPrediction < 0.5):
                    prediction = ~y
                if(prediction == y):
                    correctCount += 1    
            accuracyList.append(correctCount/len(y_cross_test))   
        averageAccuracy = sum(accuracyList)/5  
        print("Average accuracy = {}".format(averageAccuracy))
        if(averageAccuracy > optimalAccuracy):
            optimalW = mean  
            optimalAccuracy = averageAccuracy
    return optimalW


def MCMC(w, x_train, y_train, posteriorVariance, variance):
    wStar = proposedPDF(w, posteriorVariance)
    wStar = np.squeeze(wStar)
    
    wStarList = dataLikelihood(wStar, x_train, y_train)
    priorWStar = priorTerm(wStar, variance)
    star = wStarList * priorWStar

    w = np.squeeze(w)
    wLikelihood = dataLikelihood(w, x_train, y_train)
    priorW = priorTerm(w, variance)
    
    posterior = wLikelihood * priorW

    threshold = np.minimum(star/posterior, 1)
    if np.random.uniform(low=0.,high=1.) < threshold:
        pass
    else:
        wStar = w
    return wStar


def crossValidationMCMC(x_folds, y_folds, wList, qDistribution, variance):
    accuracyList = []
    print("\nMCMC Cross Validation:\n")
    for j in range(5):
        print("Fold = {}".format(j + 1))
        #Initializing: 
        correctCounter = 0
        # Creating folds:
        x_fold, y_fold = x_folds[j], y_folds[j]
        y_cross_train, y_cross_test = np.array(y_fold[0]), np.array(y_fold[1])
        x_cross_train, x_cross_test = np.array(x_fold[0]), np.array(x_fold[1])

        # Generating predictions:
        posteriorPredictionList = posteriorPredictor(x_cross_train, y_cross_train, x_cross_test, y_cross_test, wList, qDistribution, variance)
        
        # Cycling through predictions:
        for i in range(len(posteriorPredictionList)):
            posteriorPrediction = np.squeeze(posteriorPredictionList[i])
            y = np.squeeze(y_cross_test[i])  
            prediction = y
            if(posteriorPrediction < 0.5):
                prediction = ~y
            if(prediction == y):
                correctCounter += 1   
        accuracyList.append(correctCounter/len(y_cross_test))   
    averageAccuracy = sum(accuracyList)/5 
    print("Average accuracy = {}".format(averageAccuracy))
    return averageAccuracy

def posteriorPredictor(x_train, y_train, x_test, y_test, wProposed, qDistribution, variance):
    posteriorPredictionList= []
    j = 0
    # Looping through all test points:
    for j in range(len(x_test)):
        # Initializing normalization term, r list, and test data:
        normalizationTerm = 0
        rList = []    
        x = x_test[j]
        y = y_test[j]
        
        # Looping through proposed weights:
        for j in range(np.shape(wProposed)[0]):
            p = dataLikelihood(wProposed[j], x_train, y_train) * priorTerm(wProposed[j], variance)
            q = qDistribution[j]            
            r = p/q
            rList.append(r)
            normalizationTerm += r

        # Resetting posterior prediction to 0:
        posteriorPrediction = 0
        for j in range(np.shape(wProposed)[0]): 
            wLikelihood = eachLikelihood(wProposed[j],x,y)
            posteriorPrediction += wLikelihood*rList[j]/normalizationTerm
        posteriorPredictionList.append(posteriorPrediction)   
        
    return posteriorPredictionList


def laplaceApproximation(x_train, x_test, y_train, y_test, learningRate, variance, maxIterations=inf):
    # Initializing:
    x_train = createMatrix(x_train)
    x_test = createMatrix(x_test)
    w = np.zeros(np.shape(x_train[0]))
    iterations = 0
    result = np.reshape(np.dot(x_train, w), np.shape(y_train))
    gradient = likelihoodGradient(x_train, result, y_train) + priorGradient(w, variance)

    # Gradient descent:
    while max(gradient) > 0.00000000001 and iterations < maxIterations:
        result = np.dot(x_train, w)
        gradient = likelihoodGradient(x_train, result, y_train) + priorGradient(w, variance)
        w = np.add(w, learningRate * gradient)
        iterations += 1

    H = likelihoodGradient2(x_train, result) + priorGradient2(w, variance)
    margLikelihood = logLikelihood(result, y_train) + priorLog(w, variance) - logG(H)

    return margLikelihood, iterations, w, H

def question1a():
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
    y_train, y_valid, y_test = y_train[:, (1,)], y_valid[:, (1,)], y_test[:, (1,)]
    learningRate = 0.0001
    maxIterations = 1000
    x_train, x_test = np.vstack((x_train, x_valid)), x_test
    y_train, y_test = np.vstack((y_train, y_valid)), y_test
    varianceList = [0.5, 1, 2]
    print("\nResults for question 1:\n")
    for variance in varianceList:
        margLikelihood, iterations, w, H = laplaceApproximation(x_train, x_test, y_train, y_test, learningRate, variance, maxIterations)
        print("For a variance of {}:".format(variance))
        print("Iterations = {}".format(iterations))
        print("Marginal log likelihood = {}\n".format(margLikelihood))

def question1b():
    # Load dataset to calculate the marginal likelihood using the functions from question 1a:
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
    y_train, y_valid, y_test = y_train[:, (1,)], y_valid[:, (1,)], y_test[:, (1,)]

    # Initialization:
    # Use a variance of 1 for question 1b:
    variance = 1
    correctCount = 0
    learningRate = 0.001
    maxIterations = 1000

    # Using functions from question 1a to calculate marginal likelihood, iterations, w, and H:
    margLikelihood, iterations, w, H = laplaceApproximation(np.vstack((x_train, x_valid)), x_test,\
       np.vstack((y_train, y_valid)), y_test, learningRate, variance, maxIterations)


    # Reload dataset for question 1b:
    x_train, x_valid, x_test, y_train, y_valid, y_test = list(load_dataset('iris'))
    x_train, x_test = np.vstack((x_train, x_valid)), x_test
    y_train, y_test = np.vstack((y_train[:,(1,)], y_valid[:,(1,)])), y_test[:,(1,)]
    x_train = np.append(np.ones((np.shape(x_train)[0], 1)), x_train, axis = 1)
    x_test = np.append(np.ones((np.shape(x_test)[0], 1)), x_test, axis = 1)
    w = np.concatenate(np.reshape(w, (1, 5)))

    # Calculating proposed w using the multivariateNormal function:
    wProposed = multivariateNormal(mean=w, cov=np.linalg.inv(-H)*variance, size=10000)
    # Using a q distribution as proposed:
    qDistribution = multivariate_normal.pdf(wProposed, mean=w, cov=np.linalg.inv(-H)*variance) 


    # Visualizing the data:
    priorList = []
    likelihoodList = []
    posteriorList= []

    # Generating plots:
    w0 = [element[0] for element in wProposed]
    w1 = [element[1] for element in wProposed]
    w2 = [element[2] for element in wProposed]
    w3 = [element[3] for element in wProposed]
    w4 = [element[4] for element in wProposed]

    # Looping through the weights:
    for w in wProposed:
        likelihoodList.append(dataLikelihood(w,x_train,y_train))
        priorList.append(priorTerm(w,variance))
        
    dimension = np.shape(wProposed)[0]
    priorList = np.reshape(np.array(priorList), (dimension, 1))
    likelihoodList = np.reshape(np.array(likelihoodList), (dimension, 1))
    posteriorList = (likelihoodList*priorList)/(np.sum(likelihoodList * priorList)/(np.shape(wProposed)[0]))
    
    priorList = list(priorList)
    likelihoodList = list(likelihoodList)
    posteriorList = list(posteriorList)
    
    posteriorList = posteriorList[0::10]
    priorList = priorList[0::10]
    qDistribution = qDistribution[0::10]
    w0 = w0[0::10]
    w1 = w1[0::10]
    w2 = w2[0::10]
    w3 = w3[0::10]
    w4 = w4[0::10]

    fig, ax = plt.subplots(3,2)
    ax[0][0].plot(w0, posteriorList, 'bo', alpha = 0.8, markersize = 3, label = 'Posterior')
    ax[0][0].plot(w0, qDistribution, 'go', alpha = 0.8, markersize = 3, label = 'Proposed')
    ax[0][0].set_xlabel('w0')                            
    ax[0][0].legend()                                    
                                                        
    ax[0][1].plot(w1, posteriorList, 'bo', alpha = 0.8, markersize = 3, label = 'Posterior')
    ax[0][1].plot(w1, qDistribution, 'go', alpha = 0.8, markersize = 3, label = 'Proposed')
    ax[0][1].set_xlabel('w1')              
    ax[0][1].legend()                      
                                                
    ax[1][0].plot(w2, posteriorList, 'bo', alpha = 0.8, markersize = 3, label = 'Posterior')       
    ax[1][0].plot(w2, qDistribution, 'go', alpha = 0.8, markersize = 3, label = 'Proposed')
    ax[1][0].set_xlabel('w2')              
    ax[1][0].legend()                      
                                           
    ax[1][1].plot(w3, posteriorList, 'bo', alpha = 0.8, markersize = 3, label = 'Posterior')       
    ax[1][1].plot(w3, qDistribution, 'go', alpha = 0.8, markersize = 3, label = 'Proposed')
    ax[1][1].set_xlabel('w3')              
    ax[1][1].legend()                      
                                                      
    ax[2][0].plot(w4, posteriorList, 'bo', alpha = 0.8, markersize = 3, label = 'Posterior')
    ax[2][0].plot(w4, qDistribution, 'go', alpha = 0.8, markersize = 3, label = 'Proposed')
    ax[2][0].set_xlabel('w4')              
    ax[2][0].legend() 
    fig.set_size_inches(19, 10.5)
    fig.delaxes(ax[2][1])
    fig.savefig('results/dataVisualization.png')
    fig.show()
    # End of data visualization


    # Choosing w using 5 fold cross-validation:
    # Vary the mean:
    possibleMeanValues = [w - 1, w - 0.5, w, w + 0.5, w + 1]
    w = crossValidation(x_train, y_train, possibleMeanValues, H, variance) 
   
    wProposed = multivariateNormal(mean=w, cov=np.linalg.inv(-H)*variance, size=1000)
    qDistribution = np.reshape(multivariate_normal.pdf(wProposed, mean=w, cov= np.linalg.inv(-H)*variance) , (1000, 1))

    posteriorPrediction = posteriorPredictor(x_train, y_train, x_test, y_test, wProposed, qDistribution, variance)

    for j in range(len(posteriorPrediction)):
        predictionElement = np.squeeze(posteriorPrediction[j])
        y = np.squeeze(y_test[j])
        prediction = y
        if(predictionElement < 0.5):
            prediction = ~y 
        print("Prediction = {}, Actual = {}".format(prediction, np.squeeze(y_test[j]))) 
        if prediction == y_test[j]:
            correctCount += 1        
    print("Accuracy = {}".format(correctCount/len(y_test)))
    return

def question1c():
    # Intializing data for computation of marginal likelihood from question 1a:
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
    y_train, y_valid, y_test = y_train[:, (1,)], y_valid[:, (1,)], y_test[:, (1,)]
    baseVariance = 1
    correctCount = 0
    learningRate = 0.001
    maxIterations = 1000
    margLikelihood, iterations, w, H = laplaceApproximation(np.vstack((x_train, x_valid)), x_test,\
        np.vstack((y_train, y_valid)), y_test, learningRate, baseVariance, maxIterations)
    
    # Reinitializing data:
    x_train, x_valid, x_test, y_train, y_valid, y_test = list(load_dataset('iris'))
    x_train, x_test = np.vstack((x_train, x_valid)), x_test
    del x_valid
    y_train, y_test = np.vstack((y_train[:,(1,)], y_valid[:,(1,)])), y_test[:,(1,)]
    del y_valid
    x_train = np.append(np.ones((np.shape(x_train)[0], 1)), x_train, axis = 1)
    x_test = np.append(np.ones((np.shape(x_test)[0], 1)), x_test, axis = 1)
    x_y_train = np.hstack((x_train, y_train))
    np.random.shuffle(x_y_train)


    # Initializing variables:
    baseVariance = 1. 
    varianceList = [0.25, 0.5, 0.75, 1, 1.25, 1.5] 
    optimalAccuracy = -1
    optimalVariance =  None
    
    
    # Looping through each variance:
    for varianceElement in varianceList:
        print("\n\nFor proposal variance:", varianceElement)
        currrentW = np.array(w)
        currentWList = np.array([[],[],[],[],[]])
        currentWStarList = np.array([[],[],[],[],[]])

        # "Burn-in" 1000 iterations:
        for j in range(1000):
            currrentW = np.reshape(MCMC(currrentW, x_train, y_train, varianceElement, baseVariance), (5,1))
            
        for i in range(10000):
            currentWStar = np.reshape(MCMC(currrentW, x_train, y_train, varianceElement, baseVariance), (5,1))
            # "Thin" by collecting every 100th sample:
            if i % 100 == 0:
                currentWStarList = np.hstack([currentWStarList, currentWStar])
                currentWList = np.hstack([currentWList, currrentW])

            currrentW = currentWStar
       
        currentWStarList = currentWStarList.T
        currentWList = currentWList.T
        
        # Initializing data for cross validation:
        x_c_train = x_y_train[:,:5]
        y_c_train = x_y_train[:,5]
        y_c_train = np.array(y_train, dtype = bool)
        y_c_train = np.expand_dims(y_train, axis = 1)
        x_c_train = list(x_train)
        y_c_train = list(y_train)
        x_c_folds = generateFolds(x_train)
        y_c_folds = generateFolds(y_train)
        qDistribution = []
        for currentWStar in currentWStarList:
            qDistribution.append(priorTerm(currentWStar, varianceElement))
            
        averageAccuracy = crossValidationMCMC(x_c_folds, y_c_folds, currentWList, qDistribution, varianceElement)
        
        if averageAccuracy > optimalAccuracy:
            optimalAccuracy = averageAccuracy
            optimalVariance = varianceElement
        
    print("Optimal variance = {}".format(optimalVariance))

    qDistribution = []
    for currentWStar in currentWStarList:
        qDistribution.append(priorTerm(currentWStar, optimalVariance))
    
    posteriorPredictionList = posteriorPredictor(x_train, y_train, x_test, y_test, currentWList, qDistribution, optimalVariance)
    
    correctCount = 0
    for i in range(len(posteriorPredictionList)):
        y = np.squeeze(y_test[i])
        prediction = y
        if np.squeeze(posteriorPredictionList[i]) < 0.5:
            prediction = ~y           
        print("Predicted = {}, Actual = {}".format(prediction, np.squeeze(y_test[i])))    
        if(prediction == y_test[i]):
            correctCount += 1        
    
    accuracy = correctCount/len(y_test)
    print("Accuracy = {}".format(accuracy))
    
    flower9List = []
    flower10List = []

    for weight in currentWList:        
        flower9List.append(eachLikelihood(weight, x_test[8], True))            
    
    for weight in currentWList:
        flower10List.append(eachLikelihood(weight, x_test[9], True))  
 
    flower9List = np.squeeze(flower9List)
    flower10List = np.squeeze(flower10List)

    fig, ax = plt.subplots()
    plt.hist(flower9List, color = "m", alpha=0.8, label = "9th Flower")
    plt.hist(flower10List, color = "c", alpha = 0.8, label = "10th Flower")
    plt.title('Histogram of 9th and 10th Flowers')
    plt.xlabel("Probability")
    plt.ylabel("Number of Occurrences")
    plt.legend()
    fig.savefig("results/Histogram.png")

   
    return



if __name__ == "__main__":
    #question1a()
    #question1b()
    #question1c()
    