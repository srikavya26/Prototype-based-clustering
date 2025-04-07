# Prototype-Based Clustering: K-Means vs. Fuzzy C-Means

## Introduction

This report explores the comparison between two prototype-based clustering algorithms: **K-Means** and **Fuzzy C-Means (FCM)**, applied to the **Sales Transactions Weekly dataset**. The goal is to evaluate and compare their performance based on various clustering metrics, providing insights that can guide business decisions.

## Clustering Algorithms

### K-Means
K-Means is a widely-used clustering algorithm that partitions the data into `K` clusters, each defined by its centroid. The algorithm is based on minimizing the sum of squared distances between data points and their respective cluster centroids.

### Fuzzy C-Means (FCM)
Fuzzy C-Means is an extension of K-Means that allows data points to belong to multiple clusters with varying degrees of membership. Instead of assigning each data point to a single cluster, FCM assigns a probability (membership value) to each point for each cluster.

## Methodology

### Dataset
The **Sales Transactions Weekly dataset** contains transactional data, including attributes such as sales amounts, dates, and product categories. The dataset consists of `811` observations and `107` attributes.

### Preprocessing
- The dataset was cleaned and normalized before applying clustering algorithms.
- Missing values were handled by imputation methods.

### Clustering Execution
- **K-Means** was executed with `K=2` based on domain knowledge and a preliminary analysis of the data.
- **Fuzzy C-Means** was applied with a fuzzification parameter of `m=2` to control the degree of membership.

### Evaluation Metrics
Several clustering evaluation metrics were used to compare the results of both algorithms:
- **Silhouette Score**: Measures the quality of clustering by evaluating how similar an object is to its own cluster compared to other clusters.
- **Davies-Bouldin Index**

### Comparison of Results
The results suggest that **Fuzzy C-Means** outperforms K-Means in terms of cluster compactness and quality. Fuzzy C-Means provided better separation between clusters, as evidenced by the higher silhouette score.

## Business Insights

- **K-Means** clusters may be too rigid for certain business use cases, where data points exhibit mixed characteristics.
- **Fuzzy C-Means** can be more suitable for cases where transactions might belong to multiple categories, offering more flexibility in categorizing sales transactions.

## Conclusion

In conclusion, Fuzzy C-Means is a more flexible approach to clustering sales transactions, as it accommodates the possibility that each transaction may exhibit characteristics of multiple clusters. While K-Means is a simpler and faster algorithm, FCM provides a more nuanced understanding of the data, making it a better choice for more complex datasets with overlapping characteristics.

## References

- Introduction to Data Mining by Tan et al (Second edition) is Chapter 8 Section 2 (pp. 641-657)
  
## Future Work

Future improvements could involve:
- Testing with different numbers of clusters.
- Using other clustering algorithms like DBSCAN or Hierarchical Clustering.
- Exploring different feature engineering techniques to improve clustering results.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
