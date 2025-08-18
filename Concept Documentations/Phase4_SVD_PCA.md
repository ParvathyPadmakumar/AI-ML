# SVD_PCA

## Documentation

- used terminal command to detect file encoding: file -i alt.atheism.txt
alt.atheism.txt: message/rfc822; charset=iso-8859-1
- labelled all lines as 0 initially for cluster plotting since they didn't have proper groups
- implemented tf-idf and svd dimension reduction(to 2)
- calculated silhoutte score
- plotted cluster plot with y label prediction after kmeans clustering and before  

tf-idf: Converts text into high-dimensional sparse numeric matrix

Step after TF-IDF to reduce dimensions:

- SVD (TruncatedSVD) or
- PCA or
- t-SNE/UMAP

## Questions

- Why dimensionality reduction is used (mention sparsity of TF-IDF)

The TF-IDF matrix will have one column for each unique word. Most columns will be zero as some words will only be important in specific documents. Sparse matrix means it has mostly 0 values.So, after TF-IDF we have a high-dim sparse matrix. Hence PCA/SVD is used to reduce dimension.

- What SVD/PCA achieves

It extracts the topic structure and hence, groups similar patterns. Here, i have also used it to reduce dimension into 2D for plotting.

- How clustering helps understand the structure of data without labels

It helps group similar data together which can be further plotted to analyze the structure of the data. The predicted points act as labels.
