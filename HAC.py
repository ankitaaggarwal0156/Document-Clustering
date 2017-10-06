import sys
import heapq
from scipy.sparse import lil_matrix
import numpy as np
import math
from scipy.sparse.linalg import norm

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
    inner = A.multiply(B)
    k1 = norm(A)
    k2 = norm(B)
    t = inner.sum()/(k1*k2)
    return t

def Pairdistance(d1, mat):
    distance=[]
    for i in xrange(d1-1):
        for j in range(i+1, d1):
            c = Cosine(mat[i], mat[j])
            dis = float(1-c)
            distance.append((dis,[dis,[[i],[j]]]))
    return distance

def buildQueue(dis_list):
    heapq.heapify(dis_list)
    heap = dis_list
    return heap         

def hierarchialClustering(n,d1,mat):
    cluster1={}
    old_clusters = []
    for i in xrange(d1):
        cluster1key = str([i])
        cluster1.setdefault(cluster1key, {})
        cluster1[cluster1key].setdefault("centroid", mat[i])
        cluster1[cluster1key].setdefault("elements", [i])
    current_clusters=cluster1
    while(len(current_clusters)>n):
        d,m1 = heapq.heappop(heap)
        p= m1[1]
        if not checkValid(m1, old_clusters):
            continue
        new_cluster = {}
        new_cluster_elements = sum(p, [])
        new_cluster_center = computeCentroid(mat, new_cluster_elements)
        new_cluster.setdefault("centroid", new_cluster_center)
        new_cluster.setdefault("elements", new_cluster_elements)
        for item in p:
            old_clusters.append(item)
            del current_clusters[str(item)]
        PushHeap(heap, new_cluster, current_clusters, mat)
        current_clusters[str(new_cluster_elements)] = new_cluster
    return current_clusters.values()
        
def checkValid(heap_node, old_clusters):
    pdata = heap_node[1]
    for o in old_clusters:
        if o in pdata:
            return False
    return True


def computeCentroid(mat, index):
    size = len(index)
    s=0.0
    for i in index:
        s = s+ mat[i]
    centroid =s/size
    return centroid
       
def PushHeap(heap, new_cluster, current_clusters,mat):
    jj = new_cluster["centroid"]
    for e in current_clusters.values():
        heap1 = []
        c= Cosine(e["centroid"],jj)
        dist= float(1-c)
        heap1.append(dist)
        heap1.append([new_cluster["elements"], e["elements"]])
        heapq.heappush(heap, (dist, heap1))
        
filename = sys.argv[1]
n = int(sys.argv[2])
mat,d1,w1 = ReadFile(filename)
heap=[]
heap=Pairdistance(d1, mat)
heap=buildQueue(heap)
c= hierarchialClustering(n,d1,mat)
for cluster in c:
    cluster["elements"].sort()
    print( ", ".join( repr(int(e)+1) for e in cluster["elements"] ) )