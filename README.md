# CIA CREST Archive Explorer (cia-crest-explorer)
Exploring the CIA CREST Archive (https://www.cia.gov/library/readingroom/collection/crest-25-year-program-archive) through Document Clustering.

## 1. Introduction
Welcome! The CREST Explorer is a tool that allows you to OCR and cluster documents. In particular, we focus on the CIA CREST Archive of declassified documents.

To get started, we OCRed ~1000 documents. You are welcome to add more documents from the archive or correct the OCR errors on the current documents. If you do, please submit a pull request, we would love to get as much help as we can :)

Below you can find a breakdown of the tool and a guide on how to use it.


## 2. How to use this tool
You will find three folders and a script in the main project folder.

1. dialog.py

Run this script for more specific instructions about each of the tools described here.

2. clustering/

In this folder you will find the `crest_clustering.py` script that allows you to run a machine learning clustering algorithm to group the dataset documents in clusters. The clustering occurs by weighted term frequency. In this folder you can also see a sample output `clusters_40/`, which is a sample run of the clustering script with a number of clusters set to 40. This can be modified in the script.
If you are unsure of the number of clusters to choose, you may try to use the Silhoutte Method described [here](http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py).

3. dataset/

Inside of the dataset folder you can find ~1000 text files. These are files that contain the text extracted from the original PDFs on the CIA CREST Archive. Inside of the folder, you can find subfolders that break down the documents by the collection they belong to. For more information on the available collections, please refer to the CREST Archive website noted above.

4. tools/

The tools folder contains three scripts to OCR, create XML files to be fed to a Solr search engine, and a simple implementation of a generative machine learning model to run with the CREST Archive documents.
To OCR a folder with documents, please refer to the `ocr_tool/convert_all.sh` script. Or you may simply run the `get_text.py` script. Note that *this script takes in an input file as an argument*. The OCR tool we are using here is [doc2text](https://github.com/jlsutherland/doc2text).

## 3. Contact Information
[Raquel Alvarez](rva5120@cse.psu.edu)
