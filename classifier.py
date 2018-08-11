from __future__ import division
import numpy as np


def run_train_test(training_input, testing_input):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition. 
    You are you are permitted to use the numpy library but you must write 
    your own code for the linear classifier. 

    Inputs:
        training_input: list form of the training file
            e.g. [[3, 5, 5, 5],[.3, .1, .4],[.3, .2, .1]...]
        testing_input: list form of the testing file

    Output:
        Dictionary of result values 

        IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED
        
        Example:
            return {
                "tpr": true_positive_rate,
                "fpr": false_positive_rate,
                "error_rate": error_rate,
                "accuracy": accuracy,
                "precision": precision
            }
    """


    # TODO: IMPLEMENT
    # separate data into A, B, and C
    numFeat = training_input[0][0];
    numA = training_input[0][1];
    numB = training_input[0][2];
    numC = training_input[0][3];

    A = training_input[1:numA+1];
    B = training_input[numA+1:numA + numB+1];
    C = training_input[numA + numB + 1:numA + numB + numC+1];
   
    #find centroids
    centroidA = [None]*numFeat;
    centroidB = [None]*numFeat;
    centroidC = [None]*numFeat;

    for i in range(0,numFeat):
        sumA = 0;
        sumB = 0;
        sumC = 0;
        for j in range(0, numA):
            sumA +=  A[j][i];
        centroidA[i] = sumA/numA;

        for k in range(0, numB):
            sumB += B[k][i];
        centroidB[i] = sumB/numB;

        for l in range(0, numC):
            sumC += C[l][i];
        centroidC[i] = sumC/numC;

        
    
    # construct discriminant function(basic linear classifier) between classes
    #w dot x = t

    #w values w = p - n
    wAB = [None] * numFeat;
    wAC = [None] * numFeat;
    wBC = [None] * numFeat;

    for i in range(0, numFeat):
        wAB[i] = centroidA[i] - centroidB[i];
        wAC[i] = centroidA[i] - centroidC[i];
        wBC[i] = centroidB[i] - centroidC[i];

    #squared lengths of vectors
    lengthA = 0;
    lengthB = 0;
    lengthC = 0;
    for i in range(0, numFeat):
        lengthA = lengthA + (centroidA[i]**2)
        lengthB = lengthB + (centroidB[i]**2)
        lengthC = lengthC + (centroidC[i]**2)
    
    #t values
    tAB = (lengthA - lengthB)/2
    tAC = (lengthA - lengthC)/2
    tBC = (lengthB - lengthC)/2

    #keep track of tpr, fpr
    testData = testing_input.pop(0);
    testData.pop(0);
    dim = testData[0];
    #rates = [{
    #    "true_positive": 0,
    #    "true_negative": 0,
    #    "false_positive":0,
    #    "false_negative":0},
    # {
    #    "true_positive": 0,
    #    "true_negative": 0,
    #    "false_positive":0,
    #    "false_negative":0},
    # {
    #    "true_positive": 0,
    #    "true_negative": 0,
    #    "false_positive":0,
    #    "false_negative":0}]

    tpA = 0
    fpA = 0
    tnA = 0
    fnA = 0

    tpB = 0
    fpB = 0
    tnB = 0
    fnB = 0

    tpC = 0
    fpC = 0
    tnC = 0
    fnC = 0

    # use discriminant function to decide A or B then (A or B) or C
    # range 3 for A, B, C
    #check chart given in class
    for i in range(0,3):
        for j in range(0, dim*3):
            # test A or B
            x = testing_input[j]
            if(np.dot(wAB,x) >= tAB):
                #test A or C
                if(np.dot(wAC, x) >= tAC): #CLASSIFY AS A
                    if(j >= 0 and j < dim): # A
                        #update counts
                        if(i == 0): #correctly A
                            tpA += 1;
                        elif(i == 1): # incorrectly A (B)
                            tnB += 1;
                        elif(i == 2): # incorrectly A (C)
                            tnC += 1;
                    elif(j >= dim and j < dim * 2): # B
                        if(i == 0):
                            fpA +=1;
                        elif(i == 1):
                            fnB +=1;
                        elif(i == 2):
                            tnC += 1;
                    elif(j >= dim * 2 and j < dim * 3): # C
                        if(i == 0):
                            fpA += 1;
                        elif(i == 1):
                            tnB += 1;
                        elif(i == 2):
                            fnC += 1;

                else: # CLASSIFY AS C
                    if(j >= 0 and j < dim):
                        if(i == 0): #correctly A
                            fnA += 1;
                        elif(i == 1): # incorrectly A (B)
                            tnB += 1;
                        elif(i == 2): # incorrectly A (C)
                            fpC += 1;
                    elif(j >= dim and j < dim * 2): # B
                        if(i == 0):
                            tnA +=1;
                        elif(i == 1):
                            fnB +=1;
                        elif(i == 2):
                            fpC += 1;
                    elif(j >= dim * 2 and j < dim * 3): # C
                        if(i == 0):
                            tnA += 1;
                        elif(i == 1):
                            tnB += 1;
                        elif(i == 2):
                            tpC += 1;



            else: #CLASSIFY AS B
                if(np.dot(wBC, x) >= tBC):
                    if(j >= 0 and j < dim):
                        if(i == 0): 
                            fnA += 1;
                        elif(i == 1): 
                            fpB += 1;
                        elif(i == 2):
                            tnC += 1;
                    elif(j >= dim and j < dim * 2):
                        if(i == 0):
                            tnA +=1;
                        elif(i == 1):
                            tpB +=1;
                        elif(i == 2):
                            tnC += 1;
                    elif(j >= dim * 2 and j < dim * 3):
                        if(i == 0):
                            tnA += 1;
                        elif(i == 1):
                            fpB += 1;
                        elif(i == 2):
                            fnC += 1;

                else: 
                    # CLASSIFY AS C
                    if(j >= 0 and j < dim):
                        if(i == 0): 
                            fnA += 1;
                        elif(i == 1): 
                            tnB += 1;
                        elif(i == 2): 
                            fpC += 1;
                    elif(j >= dim and j < dim * 2):
                        if(i == 0):
                            tnA +=1;
                        elif(i == 1):
                            fnB +=1;
                        elif(i == 2):
                            fpC += 1;
                    elif(j >= dim * 2 and j < dim * 3):
                        if(i == 0):
                            tnA += 1;
                        elif(i == 1):
                            tnB += 1;
                        elif(i == 2):
                            tpC += 1;

    tp = tpA + tpB + tpC
    tn = tnA + tnB + tnC
    fp = fpA + fpB + fpC
    fn = fnA + fnB + fnC
    tp = tp/3;
    tn = tn/3;
    fp = fp/3;
    fn = fn/3;
    
    #get total positives and negatives
    P = testData[0]
    N = testData[0] + testData[0]
    P_hat = tp + fp

    true_positive_rate = tp/P;
    false_positive_rate = fp/N;
    error_rate = (fp + fn)/(P+N);
    accuracy = 1 - error_rate;
    precision = tp/P_hat;

    results = {
        "tpr": true_positive_rate,
        "fpr": false_positive_rate,
        "error_rate": error_rate,
        "accuracy": accuracy,
        "precision": precision,
    }

    print results

    return results

def parse_file(filename):
    """
    This function is provided to you as an example of the preprocessing we do
    prior to calling run_train_test
    """
    with open(filename, "r") as f:
        data = [[float(y) for y in x.strip().split(" ")] for x in f]
        data[0] = [int(x) for x in data[0]]

        return data

if __name__ == "__main__":
    import sys

    training_input = parse_file(sys.argv[1])
    testing_input = parse_file(sys.argv[2])

    run_train_test(training_input, testing_input)

