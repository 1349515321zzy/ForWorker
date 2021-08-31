import numpy as np
import torch 
from torch import nn

from math import sqrt
def cal(prob,labels):

    f = list(zip(prob,labels))

    rankList = [values2 for values1,values2 in sorted(f,key=lambda x:x[0])]

    rank = [i+1 for i in range(len(rankList)) if rankList[i]==1]

    pos = np.sum(labels==1)
    neg = np.sum(labels==0)

    auc = (np.sum(rank) - (pos*(pos+1)/2))/ (pos *neg)

def cross_entropy_loss(pre,true):
    delta = 1e-7

    return -np.sum(true * np.log(pre + delta))



def multi_head(q,k,v,num_heads,dim_k,dim_v,attn_mask):

    batch , seq, dims_in = q.shape

    dk = dim_k//num_heads
    dv = dim_v//num_heads


    q = nn.Linear(dims_in,dim_k)(q).reshape(batch, seq, num_heads,dk).transpose(1,2)
    k = nn.Linear(dims_in,dim_k)(k).reshape(batch, seq, num_heads,dk).transpose(1,2)

    v = nn.Linear(dims_in,dim_k)(v).reshape(batch, seq, num_heads,dv).transpose(1,2)

    attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1) 
    dist = dist.masked_fill(attn_mask == 0, -1e9)

    dist = torch.matmul(q,k.tranpose(2,3)) * (1/sqrt(dk))
    dist = torch.softmax(dist,dims=-1)

    att = torch.matmul(dist,v)
    att.transpose(1,2).reshape(batch,n,dim_v)

    return att


def initCentroids(dataSet, k):
	numSamples, dim = dataSet.shape
	centroids = zeros((k, dim))
	for i in range(k):
		index = int(random.uniform(0, numSamples))
		centroids[i, :] = dataSet[index, :]
	return centroids
 
def kmeans(datset, k):
    numSamples = datset.shape[0]
    # first column stores which cluster this sample belongs to,
	# second column stores the error between this sample and its centroid
    clusterAs =  mat(zeros(numSamples,2))

    isChange = True
    centroids = initCentroids(dataSet, k)

    while isChange:

        isChange = False

        for i in xrange(clusterAs):
            minDis = 100000.0
            minIndex = 0
            ## step 2: find the centroid who is closest
            for j in range(k):

                distance = sqrt(sum(pow(centroids[j,:]-datset[i,:],2)))
                if(distance<minDis):
                    minIndex = j
                    minDis = distance
            ## step 3: update its cluster
            if clusterAs[i,0]!= minIndex:
                isChange = True
                clusterAs[i,:] = minIndex, minDis**2
        
        # step 4: update centroids
        for j in range(k):
			pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
			centroids[j, :] = mean(pointsInCluster, axis = 0)
    return centroids, clusterAssment

    

   