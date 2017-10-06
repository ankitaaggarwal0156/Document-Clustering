# Document-Clustering
Hierarchial Agglomerative clustering , K-Means implementation for Document clustering in Python Spark

Workaround:
Documents are represented as a unit vector. The unit vector of a document is obtained from tf x idf  vector of the document, normalized (divided) by its Euclidean length. tf is term frequency (# of occurrences of word in the document) and idf is given by log( 𝑁+1/ df+1) , where N is the number of documents in the given collection and df is the number of documents where the word appears.

Similarity/ distance is measured using Cosine function.

## HAC
<li> Used a heap based prioririty queue to store pairwise distance of clusters. </li>
<li> Nodes are removed from heap using a delayed deletion, instead of going for an O(n^2) deletion approach. </li>

## Execution format:
python hac.py docword.txt k

The program takes 2 arguments:
<li> docword.txt is a document-word file. We will describe its format in the Data Sets section.</li>
<li> k is the desired number of clusters. </li>

## Output:
For each cluster, output shows documents IDs that belong to this cluster in one line. Each ID is separated by comma. For example,
96,50
79,86,93
97
4,65,69,70
…

## Spherical K-Means:
Modified pre-existing Spark installation K-means code to account for document vectors.

<li> To obtain initial centroids, document vector RDD is sort in the increasing order of the document ID.(as in input file) </li>
<li> Using rdd.repartition(1).takeSample(False, k, 1) on the SORTED RDD, to obtain initial centroids. </li>

## Execution Format
bin/spark-submit k-means.py inputfile k convergeDist output.txt
The program takes 4 arguments:
<li> inputFile is a document-word file. We will describe its format in Data Sets section. </li>
<li> k is the number of initial points we get. </li>
<li> convergeDist is the given threshold. We use this float number as the k-means stopping criterion. </li>
<li> output.txt is the output file. </li>


