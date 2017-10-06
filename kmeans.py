from __future__ import print_function
import sys
from scipy.sparse import lil_matrix
import numpy as np
import math
from pyspark import SparkContext
sc = SparkContext(appName="PythonKmeans")

def ReadFile(filename):
    with open(filename, "r") as fo:
        doc1 = int(fo.next().strip())
        wordVocab = int(fo.next().strip())
        words = int(fo.next().strip())
        mat = lil_matrix((doc1,wordVocab))
        for line in fo:
            a=[]
            a=line.split()
            mat[int(a[0])-1,int(a[1])-1] = int(a[2])
    fo.close()
    id1=list()
    for i in xrange(wordVocab):
        df =mat[:,i].size
        id1.append(idf(doc1, df))
    mat = mat.tocoo()
    mat1 = lil_matrix((doc1,wordVocab))
    for i,j,v in zip(mat.row, mat.col, mat.data):
        mat1[i,j]= float(v*id1[j])
        
    docLength=list()
    for i in xrange(doc1):
        e = mat1[i,:].toarray().dot(mat1[i,:].toarray().T)
        e = np.sqrt(e)
        docLength.append(e)
    for i,j,v in zip(mat.row, mat.col, mat.data):
        mat1[i,j]= mat1[i,j]/docLength[i]
    
    mat1=mat1.tocsc()
    return mat1, doc1, wordVocab
    
 
def idf(doc1, df):
    return math.log((float(doc1+1)/float(df+1)),2)
            
def Cosine(A,B):
    inner = np.dot(A,B.T)
    k1= np.sqrt(np.dot(A,A.T))
    k2= np.sqrt(np.dot(B,B.T))
    t= inner/(k1*k2)
    return 1-t

def closestPoint(p, centers):
    bestIndex = 0
    closest = float("+inf")
    for i in range(len(centers)):
        tempDist = Cosine(p[1].toarray(), centers[i][1].toarray())
        if tempDist < closest:
            closest = tempDist
            bestIndex = i
    return bestIndex

if __name__ == "__main__":

    if len(sys.argv) != 5:
        print("Usage: kmeans <file> <k> <convergeDist> <output file>", file=sys.stderr)
        exit(-1)
    filename = "./"+sys.argv[1]
    mat,d1,w1 = ReadFile(filename)
    s={}
    for i in xrange(d1):
        s[i]=mat[i,:]
    data = sc.parallelize((k,v) for k,v in s.items())
    data = data.sortByKey().cache()
    K = int(sys.argv[2])
    convergeDist = float(sys.argv[3])
    kPoints = data.repartition(1).takeSample(False, K, 1)
    tempDist = 1.0
    while tempDist > convergeDist:
        closest = data.map(
            lambda p: (closestPoint(p, kPoints), (p[1], 1)))
        pointStats = closest.reduceByKey(
            lambda p1_c1, p2_c2: (p1_c1[0]+ p2_c2[0], p1_c1[1] + p2_c2[1]))
        newPoints = pointStats.map(
            lambda st: (st[0], st[1][0] / st[1][1])).collect()
        tempDist = sum(np.sum((kPoints[iK][1].toarray() - p.toarray()) ** 2) for (iK, p) in newPoints)
        for (iK, p) in newPoints:
            kPoints[iK] = (iK,p)
    outp = sys.argv[4]
    f = open(outp, "w")
    for i in range (0,K):
        f.write("%s\n" %str(kPoints[i][1].size))
    f.close()