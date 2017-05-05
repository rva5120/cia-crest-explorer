from __future__ import print_function

#from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser

import sys
import os
import shutil
from time import time

import numpy as np


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')



##############
# Parameters #
##############
number_of_clusters = 40
number_of_features = 10000
##############

##########################
# COMMAND LINE ARGUMENTS #
##########################
### Not using this right now, but might come in handy later
# Parse command line arguments
op = OptionParser()
op.add_option("--lsa", dest="n_components", type="int", help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch", action="store_false", dest="minibatch", default=True, help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf", action="store_false", dest="use_idf", default=True, help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing", action="store_true", default=False, help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000, help="Maximum number of features (dimensions) to extract from text.")
op.add_option("--verbose", action="store_true", dest="verbose", default=False, help="Print progress reports inside k-means algorithm.")

#print(__doc__)
#op.print_help()

# Only options are valid as arguments
(opts, args) = op.parse_args()
if len(args) > 0:
	op.error("this script takes no arguments.")
	sys.exit(1)




##########################
# DATASET 		 #
##########################
# Load dataset
# To load files, the directory passed needs to have the following structure:
#	text_only
#		folder1
#			file1
#			file2
#			...
#		folder2
#		...
dataset = load_files("/home/grads/rva5120/cia-crest-explorer/dataset")
print(dataset.filenames)

# Print Dataset Statistics
print("%d documents" % len(dataset.data))
#print("%d categories" % len(dataset.target_names))
print()

# Set the number of clusters for the model
true_k = number_of_clusters

# Set the number of features to extract
n_features = number_of_features

# Extract features:
# 	- TF-IDF method (defaults to IDF unless otherwise specified by the user)
#	- Removing engilsh stop words
#	- Extracting a max of n_features
print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
vectorizer = TfidfVectorizer(max_features=n_features, stop_words='english', use_idf=opts.use_idf)

''' --- not in use, here just for reference if needed in the future
if opts.use_hashing:
	if opts.use_idf:
		# Perform an IDF normalization on the output of HashingVectorizer
		hasher = HashingVectorizer(n_features=opts.n_features, stop_words='english', non_negative=True, norm=None, binary=False)
		vectorizer = make_pipeline(hasher, TfidfTransformer())
	else:
		vectorizer = HashingVectorizer(n_features=opts.n_features, stop_words='english', non_negative=False, norm='l2', binary=False)
else:
	vectorizer = TfidfVectorizer(max_features=opts.n_features, stop_words='english', use_idf=opts.use_idf)
'''

# Fit transform	  learn vocabulary and idf, return term-document matrix
# X		  term-document matrix (n_documents, n_features)
X = vectorizer.fit_transform(dataset.data)

# Print stats: computation time, and number of samples and features
print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)





##################################
# LSA - Latent Semantic Analysis #
# K-Means - Clustering		 #
##################################

# We choose MiniBatches of K-Means for Clustering and TF-IDF for feature selection
'''
# SVD - Singular Value Decomposition (if chosen by the user) ---- not in use
# This reduces text data into a manageable number of dimensions for analysis.
# For LSA, 100 n_components is recommended.
if opts.n_components:
	print("Performing dimensionality reduction using LSA")
	t0 = time()
	# Vectorizer results are normalized, which makes KMeans behave as spherical k-means for better results. 
	# Since LSA/SVD results are not normalized, we have to redo the normalization.
	svd = TruncatedSVD(opts.n_components)
	normalizer = Normalizer(copy=False)
	lsa = make_pipeline(svd, normalizer)

	# Prepare input features
	X = lsa.fit_transform()
	print("done in %fs" % (time() - t0))

	# Explained Variance Ratio: percentage of variance explained by each of the selected components
	explained_variance = svd.explained_variance_ratio_.sum()
	print("Explained variance of SVD step: {}%".format(int(explained_variance * 100)))
	print()
'''


# MiniBatches of KMeans++ (default)
# 	n_clusters	true_k 		(default 8)	Num of clusters and centroids to generate.
#	init		k-means++	(default k-means++)
#	n_init		1		(default 10) 	Num of times the k-means algo. will be run 
#							with the different centroid seeds. 
#							The final result will be the best output 
#							of n_init consecutive runs in terms of inertia.
#	init_size	1000		(optional)	Max num of iterations over the complete dataset
#							before stopping independently of any stopping
#							criteria heuristics.
#	batch_size	1000		(default 100)	Size of mini batches.
if opts.minibatch:
	km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1, init_size=1000, batch_size=1000, verbose=opts.verbose)
else:
	km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1, verbose=opts.verbose)


# Run KMeans Fitting
print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("Done in %0.3fs" % (time() - t0))
print()


# Print stats
#print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
#print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
#print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
#print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, km.labels_))
print ("------------ STATISTICS ------------")
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, km.labels_, sample_size=1000))
print()






######################
# CLUSTERS	     #
######################
# doc_label	an array that contains the document number and its corresponding label/cluster
doc_label = np.zeros([len(km.labels_),2])
for idx in xrange(len(km.labels_)):
	doc_label[idx] = [idx, km.labels_[idx]]


print(" Number of documents per cluster ")
docs_per_cluster = np.zeros([true_k, 1])
for idx in xrange(len(km.labels_)):
	docs_per_cluster[km.labels_[idx]] = docs_per_cluster[km.labels_[idx]] + 1


print(docs_per_cluster)
print(doc_label[1009])
print()

# Find the closest document to each cluster center
closest_doc_to_cluster,_ = metrics.pairwise_distances_argmin_min(km.cluster_centers_, X)
ctr = 0
print("Closest document to each cluster:")
for closest_doc in closest_doc_to_cluster:
	print("Doc "+dataset.filenames[closest_doc]+" - Cluster "+str(ctr))
	ctr = ctr + 1


# Save documents to respective clusters
print("Creating Document Tree if necessary...")


# Make new folder for this run
folder_name = 'clusters_'+str(true_k)
path = r'/export/home2/grads/rva5120/cia-crest-explorer/clustering/'+folder_name
if not os.path.exists(path):
	os.makedirs(path)
	print("Directory created: "+path)

	# Make a folder for each cluster
	for ct in xrange(true_k):
		# Make cluster folder
		cluster_path = path+'/'+'cluster_'+str(ct)
		os.makedirs(cluster_path)
	print("Subdirectories created: "+str(true_k))

	# Copy the document to the cluster folder that it belongs to
	idx = 0
	for cluster in km.labels_:
		# If the file we are copying is the closest to the cluster center, add it again under a different name
		if dataset.filenames[idx] == dataset.filenames[closest_doc_to_cluster[cluster]]:
			src = dataset.filenames[idx]		# path to file
			dst = path+'/'+'cluster_'+str(cluster)	# path to cluster folder
			shutil.copy(src,dst)			# copy file to cluster folder
			old = dst+'/'+os.path.basename(src)	# path to file in cluster
			new = old[:len(old)-4]			# name without extension
			new = new+'_CLOSEST_TO_CENTER.txt'	# new filename
			print(old)
			print(new)
			os.rename(old,new)			# rename file
		# Copy the file
		src = dataset.filenames[idx]
		dst = path+'/'+'cluster_'+str(cluster)
		shutil.copy(src, dst)
		# Point to the next file
		idx = idx + 1

	print("Files copied")
	print("... Finished creating Document Tree.")





######################
# RESULTS 	     #
######################
if not opts.use_hashing:
	print("Top terms per cluster:")
	# If we used SVD recalculate original number of dimensions
	if opts.n_components:
		original_space_centroids = svd.inverse_transform(km.cluster_centers)
		order_centroids = original_space_centroids.argsort()[:,::-1]
	else:
		### argsort    returns an array with the indices that would sort order_centroids
		### [:, ::-1]  keep row order, reverse column order
		order_centroids = km.cluster_centers_.argsort()[:, ::-1]		

	### terms   10k features extracted from the documents
	terms = vectorizer.get_feature_names()
	top_terms_10 = "Top 10 terms for this cluster: "
	for i in range(true_k):
		print("Cluster %d:" % i, end='')
		### Print the 10 features with the highest score?
		for ind in order_centroids[i, :10]:
			print(' %s' % terms[ind], end='')
			top_terms_10 = top_terms_10+' '+terms[ind]
		print()
		# Add a file with the top terms for each cluster folder
		cluster_path = path+'/'+'cluster_'+str(i)
		top_terms_file = open(cluster_path+'/'+'cluster_'+str(i)+'_top_10_terms.txt', 'w')
		top_terms_file.write(top_terms_10)
		top_terms_file.close()

	# Add to the top terms file top 100 terms per cluster, for more information about it
	top_terms_100 = "Top 100 terms for this cluster: "
	for i in range(true_k):
		for ind in order_centroids[i, :100]:
			top_terms_100 = top_terms_100+' '+terms[ind]
		# Add a file with the top terms for each cluster folder
		cluster_path = path+'/'+'cluster_'+str(i)
		top_terms_file = open(cluster_path+'/'+'cluster_'+str(i)+'_top_100_terms.txt', 'w')
		top_terms_file.write(top_terms_100.encode('utf-8'))
		top_terms_file.close()
