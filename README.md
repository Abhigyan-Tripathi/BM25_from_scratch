# BM25-Based Text Classification with Parallel Processing

## Overview

This project implements a custom BM25-based text classification system from scratch in Python. Instead of training a machine learning classifier, the approach treats every training document as a candidate document and predicts the class of a test document based on its BM25 similarity with the training corpus.

For every test document:

1. The BM25 score is computed against every training document.
2. The top 10 most similar documents are retrieved.
3. A majority vote among these top 10 documents determines the predicted class.

This project focuses not only on implementing the BM25 ranking algorithm but also on optimizing it to handle larger datasets efficiently.

---

## Problem Statement

Given a training corpus and corresponding labels, the objective is to classify unseen documents using document similarity rather than a learned classification model.

The prediction pipeline consists of:

* Preprocessing the query document
* Computing BM25 scores against every training document
* Retrieving the top 10 most similar documents
* Predicting the majority class among the retrieved documents

While conceptually simple, this approach becomes computationally expensive as the training corpus grows.

---

## Initial Setup

The project initially used only **400 training documents** for similarity computation.

The reason was purely computational.

For every test document, BM25 scores had to be computed against every training document. As the size of the corpus increased, prediction time grew linearly.

With a larger training set, inference became slow enough to make experimentation impractical.

Using only 400 training instances significantly reduced prediction time but also limited the quality of retrieved neighbors, resulting in an **F1-score of approximately 0.38**.

Although inference was faster, the smaller corpus did not provide enough representative examples for accurate retrieval-based classification.

---

## Performance Bottlenecks

Profiling the implementation revealed three major bottlenecks.

### 1. Sequential Processing

Every test document was processed one after another on a single CPU core.

With thousands of test documents and thousands of training documents, this resulted in millions of BM25 score computations.

---

### 2. Sorting All Scores

For every test document, BM25 scores for all training documents were computed and fully sorted, even though only the top 10 highest-scoring documents were required.

Sorting the complete list introduced unnecessary computational overhead.

---

### 3. Recomputing Term Frequencies

The original implementation computed a word's term frequency by scanning the entire document every time a BM25 score was requested.

Since this operation was repeated millions of times, it became one of the largest contributors to the total execution time.

---

## Optimizations

### Parallel Processing

Since the prediction for each test document is completely independent, the workload is naturally parallelizable.

The prediction pipeline was parallelized using Python's `ProcessPoolExecutor`.

Instead of processing every test document sequentially, the test dataset was divided into chunks, with each worker process handling a subset of the queries.

This enabled multiple CPU cores to compute BM25 scores simultaneously, significantly reducing inference time.

To minimize communication overhead, each worker process was initialized only once with the BM25 index and supporting data, avoiding repeated serialization of large objects.

---

### Efficient Top-k Retrieval Using Heaps

The original implementation sorted every BM25 score before selecting the top 10 documents.

Since only the highest-scoring documents were required, full sorting was replaced with `heapq.nlargest()`.

This reduced unnecessary sorting work while producing the same prediction results.

---

### Precomputing Document Term Frequencies

Instead of scanning every document repeatedly to compute term frequencies, a term-frequency dictionary was precomputed for each document during initialization.

This changed term frequency lookup from repeatedly traversing the document to a constant-time dictionary lookup.

This optimization substantially reduced the cost of every BM25 score computation.

---

## Results

After introducing these optimizations, the system was able to efficiently use a significantly larger training corpus.

| Configuration            | Training Documents | Unique Words | F1 Score |
| ------------------------ | -----------------: | -----------: | -------: |
| Initial Implementation   |                400 |       ~3,000 |     0.38 |
| Optimized Implementation |              4,000 |      ~12,000 |     0.75 |

The increase in training data provided a much richer retrieval space, while the algorithmic optimizations ensured that the larger corpus remained computationally feasible.

---

## Key Takeaways

* Implemented the BM25 ranking algorithm from scratch.
* Built a retrieval-based document classifier without using traditional machine learning models.
* Optimized inference using multiprocessing across CPU cores.
* Reduced top-k retrieval cost using heap-based selection.
* Eliminated repeated term-frequency computation through precomputed document statistics.
* Improved classification performance from an F1-score of **0.38** to **0.75** by scaling from **400** to **4,000** training documents while maintaining practical inference times.

---

## Future Improvements

Potential extensions to this project include:

* Implementing BM25+ or BM25L variants.
* Combining BM25 with dense semantic embeddings for hybrid retrieval.
* Using Approximate Nearest Neighbor (ANN) search to further improve scalability.
* Replacing majority voting with a learning-to-rank model for final prediction.
