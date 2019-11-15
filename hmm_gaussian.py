#!/usr/bin/env python3
import numpy as np
if not __file__.endswith('_hmm_gaussian.py'):
    print('ERROR: This file is not named correctly! Please name it as Lastname_hmm_gaussian.py (replacing Lastname with your last name)!')
    exit(1)

DATA_PATH = "/u/cs246/data/em/" #TODO: if doing development somewhere other than the cycle server (not recommended), then change this to the directory where your data file is (points.dat)

def parse_data(args):
    num = float
    dtype = np.float32
    data = []
    with open(args.data_file, 'r') as f:
        for line in f:
            data.append([num(t) for t in line.split()])
    dev_cutoff = int(.9*len(data))
    train_xs = np.asarray(data[:dev_cutoff],dtype=dtype)
    dev_xs = np.asarray(data[dev_cutoff:],dtype=dtype) if not args.nodev else None
    return train_xs, dev_xs

def init_model(args,train_xs):
    n,dimensions = train_xs.shape
    
    if args.cluster_num:
        
        mus = np.zeros((args.cluster_num,2))

        for i in range(0,args.cluster_num):
            mus[i][0] = np.mean(train_xs) + np.random.rand()
            mus[i][1] = np.mean(train_xs) + np.random.rand()
        #TODO: randomly initialize clusters (lambdas, mus, and sigmas)
        if not args.tied:
            sigmas = np.array([np.eye(dimensions)] * args.cluster_num)
        else:
            sigmas = np.eye(dimensions)

        transitions = np.zeros((args.cluster_num,args.cluster_num))
        
         #transitions[i][j] = probability of moving from cluster i to cluster j
        for x in range(args.cluster_num):
            for y in range(args.cluster_num):
                transitions[x][y] = np.random.rand()

        
        transitions = transitions/np.sum(transitions,axis=1)

        #print(transitions)
        initials = np.zeros(args.cluster_num) #probability for starting in each state

        for x in range(0,args.cluster_num):
            initials[x] = 1/float(args.cluster_num)
        #TODO: randomly initialize clusters (mus, sigmas, initials, and transitions)
        #raise NotImplementedError #remove when random initialization is implemented
    else:
        mus = []
        sigmas = []
        transitions = []
        initials = []
        with open(args.clusters_file,'r') as f:
            for line in f:
                #each line is a cluster, and looks like this:
                #initial mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1 transition_this_to_0 transition_this_to_1 ... transition_this_to_K-1
                vals = list(map(float,line.split()))
                initials.append(vals[0])
                mus.append(vals[1:3])
                sigmas.append([vals[3:5],vals[5:7]])
                transitions.append(vals[7:])
        initials = np.asarray(initials)
        transitions = np.asarray(transitions)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(initials)

    #TODO: Do whatever you want to pack mus, sigmas, initals, and transitions into the model variable (just a tuple, or a class, etc.)
    model = [mus,sigmas,transitions,initials]
    #raise NotImplementedError #remove when model initialization is implemented
    return model

def forward(model, data, args):
    from scipy.stats import multivariate_normal
    from math import log
    alphas = np.zeros((len(data),args.cluster_num))
    
    log_likelihood = 0.0

    mus = model[0]
    sigmas = model[1]
    initials = model[3]
    transitions = model[2]


    for x in range(0,args.cluster_num):
        if args.tied:
            alphas[0,x] = initials[x]*multivariate_normal(mean=mus[x],cov=sigmas,allow_singular=True).pdf(data[0,:])
        else:
            alphas[0,x] = initials[x]*multivariate_normal(mean=mus[x],cov=sigmas[x],allow_singular=True).pdf(data[0,:])
    
    log_likelihood += np.log(np.sum(alphas[0,:]))
    alphas[0,:] = alphas[0,:]/np.sum(alphas[0,:])

    length = data.shape[0]

    for t in range(1,length):
        for k in range(0,args.cluster_num):
            for j in range(0,args.cluster_num):
            
                if args.tied:
                    alphas[t,k] += multivariate_normal(mean=mus[k], cov=sigmas, allow_singular=True).pdf(data[t])*alphas[t-1,j]*transitions[j,k]
                else:
                    alphas[t,k] += multivariate_normal(mean=mus[k], cov=sigmas[k]).pdf(data[t])*alphas[t-1,j]*transitions[j,k]

        log_likelihood += log(np.sum(alphas[t,:]))
        alphas[t,:]= alphas[t,:]/np.sum(alphas[t,:])

    #TODO: Calculate and return forward probabilities (normalized at each timestep; see next line) and log_likelihood
    #NOTE: To avoid numerical problems, calculate the sum of alpha[t] at each step, normalize alpha[t] by that value, and increment log_likelihood by the log of the value you normalized by. This will prevent the probabilities from going to 0, and the scaling will be cancelled out in train_model when you normalize (you don't need to do anything different than what's in the notes). This was discussed in class on April 3rd.
    #raise NotImplementedError
    return alphas, log_likelihood

def backward(model, data, args):
    from scipy.stats import multivariate_normal
    betas = np.zeros((len(data),args.cluster_num))

    length = data.shape[0]

    betas[length-1,:]= 1

    betas[length-1,:] = betas[length-1,:]/(np.sum(betas[length-1,:]))

    mus = model[0]
    sigmas = model[1]
    initials = model[3]
    transitions = model[2]


    for t in range(length-2,-1,-1):
        for k in range(0,args.cluster_num):
            for j in range(0,args.cluster_num):
                if args.tied:
                    betas[t,k] += betas[t+1,j]*transitions[k,j]*multivariate_normal(mean=mus[j], cov=sigmas).pdf(data[t+1])
                else:
                    betas[t,k] += betas[t+1,j]*transitions[k,j]*multivariate_normal(mean=mus[j], cov=sigmas[j]).pdf(data[t+1])

        betas[t,:] = betas[t,:]/np.sum(betas[t,:])

    #TODO: Calculate and return backward probabilities (normalized like in forward before)
    #raise NotImplementedError
    return betas

def train_model(model, train_xs, dev_xs, args):
    from scipy.stats import multivariate_normal
    #TODO: train the model, respecting args (note that dev_xs is None if args.nodev is True)

    mus = model[0]
    sigmas = model[1]
    initials = model[3]
    transitions = model[2]

    l = train_xs.shape[0]

    gammas = np.zeros((len(train_xs),args.cluster_num))
    eta = np.zeros((l,args.cluster_num,args.cluster_num))

    for iterations in range(0,args.iterations):
        alphas,ll = forward(model,train_xs,args)
        betas = backward(model,train_xs,args)

        gammas = alphas*betas
        gammas = gammas/np.sum(gammas,axis=1,keepdims = True)

        #### E- STEP 
        for t in range(1,l):
            for k in range(0,args.cluster_num): 
                for j in range(0,args.cluster_num):
                    if args.tied:
                        eta[t,k,j] = alphas[t-1,k]*betas[t,j]*transitions[k,j]*multivariate_normal(mean=mus[j], cov=sigmas).pdf(train_xs[t])
                    else:   
                        eta[t,k,j] = alphas[t-1,k]*betas[t,j]*transitions[k,j]*multivariate_normal(mean=mus[j], cov=sigmas[j]).pdf(train_xs[t])
        
            
            eta[t,:,:] = eta[t,:,:]/np.sum(eta[t,:,:])
        
        #print(eta)
        #### M-STEP 

        for k in range(0,args.cluster_num):
            initials[k] = gammas[0,k]
            for j in range(0,args.cluster_num):
                transitions[k,j] = np.sum(eta[:,k,j])/ np.sum(gammas[:,k])
            mus[k] = (np.dot(gammas[:,k].T,train_xs))/np.sum(gammas[:,k])
            diff = train_xs - mus[k]
            if args.tied:
                sigmas += np.dot(diff.T, gammas[:, k].reshape((l, 1))*diff)

            else:
                sigmas[k] = np.dot(diff.T,gammas[:, k].reshape((l, 1))*diff)
                sigmas[k] = sigmas[k]/np.sum(gammas[:, k])

    #raise NotImplementedError #remove when model training is implemented
    return model

def average_log_likelihood(model, data, args):
    #TODO: implement average LL calculation (log likelihood of the data, divided by the length of the data)
    #NOTE: yes, this is very simple, because you did most of the work in the forward function above
    ll = 0.0


    mus = model[0]
    sigmas = model[1]
    initials = model[3]
    transitions = model[2]

    n = data.shape[0]

    ll = forward(model,data,args)[1]


    #raise NotImplementedError #remove when average log likelihood calculation is implemented
    return ll/n

def extract_parameters(model):
    #TODO: Extract initials, transitions, mus, and sigmas from the model and return them (same type and shape as in init_model)
    initials = None
    transitions = None
    mus = None
    sigmas = None

    mus = model[0]
    sigmas = model[1]
    initials = model[3]
    transitions = model[2] 
    #raise NotImplementedError #remove when parameter extraction is implemented
    return initials, transitions, mus, sigmas

def main():
    import argparse
    import os
    print('Gaussian') #Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points')
    init_group = parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument('--cluster_num', type=int, help='Randomly initialize this many clusters.')
    init_group.add_argument('--clusters_file', type=str, help='Initialize clusters from this file.')
    parser.add_argument('--nodev', action='store_true', help='If provided, no dev data will be used.')
    parser.add_argument('--data_file', type=str, default=os.path.join(DATA_PATH, 'points.dat'), help='Data file.')
    parser.add_argument('--print_params', action='store_true', help='If provided, learned parameters will also be printed.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of EM iterations to perform')
    parser.add_argument('--tied',action='store_true',help='If provided, use a single covariance matrix for all clusters.')
    args = parser.parse_args()
    if args.tied and args.clusters_file:
        print('You don\'t have to (and should not) implement tied covariances when initializing from a file. Don\'t provide --tied and --clusters_file together.')
        exit(1)

    train_xs, dev_xs = parse_data(args)
    model = init_model(args,train_xs)
    model = train_model(model, train_xs, dev_xs, args)
    nll_train = average_log_likelihood(model, train_xs, args)
    print(nll_train)
    print('Train LL: {}'.format(nll_train))
    if not args.nodev:
        nll_dev = average_log_likelihood(model, dev_xs, args)
        print('Dev LL: {}'.format(nll_dev))
    initials, transitions, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str,a))
        print('Initials: {}'.format(intersperse(' | ')(np.nditer(initials))))
        print('Transitions: {}'.format(intersperse(' | ')(map(intersperse(' '),transitions))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '),mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '),map(lambda s: np.nditer(s),sigmas)))))

if __name__ == '__main__':
    main()