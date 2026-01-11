# **Architectural Blueprint for Scalable Recommender Systems: From Collaborative Filtering to Contextual Bandits**

## **Abstract**

The engineering of recommendation engines has transcended simple matrix operations to become a complex discipline intersecting data engineering, reinforcement learning, and software architecture. This report provides an exhaustive technical specification for constructing a dual-paradigm recommender system project. The proposed architecture integrates two distinct methodological approaches: the static, latent-factor analysis of Collaborative Filtering (CF) utilizing the MovieLens ecosystem, and the dynamic, online learning framework of Contextual Multi-Armed Bandits (CMAB) applied to the Amazon "Beauty" product dataset. This document serves as both a theoretical treatise and a practical implementation guide, addressing the specific challenges of data ingestion, pipeline automation via Makefiles, and the reproducibility crisis exacerbated by modern Python environment discrepancies. We analyze the mathematical underpinnings of Matrix Factorization and Upper Confidence Bound (UCB) algorithms, benchmarking their theoretical performance against the constraints of "small" project data. Furthermore, we provide a rigorous justification for the "5-core" data filtering technique and its implications for cold-start problem simulation.

## ---

**1\. Introduction: The Duality of Recommendation Paradigms**

The landscape of information filtering is currently dominated by two distinct yet complementary paradigms: the historical analysis of accumulated preferences (Collaborative Filtering) and the sequential optimization of real-time decisions (Bandits/Reinforcement Learning). Designing a research project that encompasses both requires a nuanced understanding not only of the algorithms themselves but of the divergent data engineering pipelines required to support them.

Collaborative Filtering (CF), particularly in its latent factor implementations, relies on the assumption of stationarity—that a user’s past preferences, represented as a vector in a low-rank matrix, are predictive of their future behavior.1 This approach essentially treats recommendation as a missing value imputation problem within a massive, sparse matrix of Users $\\times$ Items. The MovieLens dataset, a staple of the research community since the late 1990s, provides the ideal "sandbox" for this paradigm due to its relational structure and explicit rating feedback mechanism.2

In contrast, Contextual Bandits represent a shift towards online learning, where the system must balance exploration (gathering data about uncertain actions) and exploitation (capitalizing on known high-value actions).4 Unlike CF, which often requires a full retraining cycle to incorporate new data, bandit algorithms update their policy incrementally with each interaction. The Amazon "Beauty" dataset, characterized by its textual richness and vast item space, offers a fertile ground for simulating these sequential decision processes, particularly when framing the problem as a "simulated bandit" on logged data.5

This project aims to bridge these worlds by establishing a unified engineering substrate—a reproducible data pipeline—that feeds both a Scikit-Surprise-based CF model and a ContextualBandits-based RL agent. The following sections detail the architectural decisions required to achieve this integration, paying special attention to the often-overlooked complexities of environment management and data sanitization.

## ---

**2\. Data Ecosystem and Engineering Constraints**

The selection of datasets for a recommender system is never merely a choice of content; it is a choice of topology. The structural properties of the data—sparsity, density, cardinality, and metadata richness—dictate the feasibility of specific algorithms.

### **2.1 The MovieLens Latest Small Dataset**

Maintained by GroupLens Research at the University of Minnesota, the MovieLens datasets are the de facto standard for benchmarking collaborative filtering algorithms.7 For this project, the "Latest Small" version is selected, a choice that imposes specific constraints and advantages.

#### **2.1.1 Statistical Characteristics and Bias**

The "Small" dataset comprises approximately 100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users.2 While termed "small," its density is artificially high compared to production environments. A critical yet often overlooked characteristic of this dataset is its inclusion criteria: users are only included if they have rated at least 20 movies.7 This "minimum activity filter" introduces a survival bias. The algorithms trained on this data are effectively learning the preferences of "power users" or active cinephiles, rather than the general population.

This density ($100,000 / (600 \\times 9,000) \\approx 1.85\\%$) allows matrix factorization techniques to converge rapidly, often yielding Root Mean Squared Error (RMSE) values in the 0.85–0.90 range.8 In broader commercial datasets, densities often drop below 0.05%, where standard SVD without robust regularization fails. Therefore, while MovieLens Small is excellent for "hello world" implementations, one must recognize that it masks the sparsity challenges inherent in real-world deployment.

#### **2.1.2 Schema and Relational Integrity**

The dataset is distributed as a collection of CSV files, a format that is easily ingested but requires relational joining.

* **ratings.csv**: The core interaction file containing userId, movieId, rating, and timestamp.1 The ratings are explicit, ranging from 0.5 to 5.0 in 0.5 increments. The timestamp is a Unix epoch, which, while available, is typically discarded in pure matrix factorization approaches that view preferences as time-invariant.  
* **movies.csv**: This file maps the integer movieId to human-readable title and pipe-separated genres.1 This metadata is crucial not for the algorithm itself (which operates on IDs), but for the interpretability of the results.  
* **links.csv**: Provides a bridge to the "real world" by mapping MovieLens IDs to IMDb and TMDb identifiers.1

The stability of the download link is a critical engineering concern. GroupLens guarantees that while the *content* of the "Latest" datasets changes over time, the persistent URL schema https://files.grouplens.org/datasets/movielens/ml-latest-small.zip remains valid for automated retrieval.2

### **2.2 The Amazon Beauty 5-Core Dataset**

To explore Contextual Bandits, we turn to the Amazon product review data collected by the McAuley Lab at UCSD. Specifically, the "Beauty" category is selected for its manageability and rich textual features suitable for context vectorization.

#### **2.2.1 The Mathematics of "k-core" Filtering**

The dataset is designated as "5-core." This terminology refers to a k-core decomposition of the bipartite user-item graph. A k-core is a maximal subgraph where every vertex has a degree of at least $k$.9 In the context of the Amazon dataset, this means the data has been iteratively pruned until every remaining user has written at least 5 reviews, and every remaining item has received at least 5 reviews.

Why is this necessary?

1. **Sparsity Reduction:** The raw Amazon dataset is notoriously sparse, with millions of "singleton" users who bought only one item. Collaborative filtering and bandit algorithms struggle to learn from singletons.  
2. **Dense Sub-matrix:** The 5-core subset creates a dense core of interactions, ensuring that there is sufficient historical overlap between users to define similarity, and sufficient data per item to estimate reward probabilities.10  
3. **Benchmarking Stability:** It standardizes the evaluation set, ensuring researchers are comparing models on the same verifiable graph structure.

#### **2.2.2 The "Loose JSON" Challenge**

Unlike the relational CSVs of MovieLens, the Amazon data is distributed as "loose JSON" or "JSON Lines" (JSONL), typically compressed via GZIP (reviews\_Beauty\_5.json.gz).9

A standard JSON parser expects a file to start with \[ and end with \], containing a comma-separated list of objects. The Amazon file, however, contains a separate, valid JSON object on each line, separated by newlines.

JSON

{"reviewerID": "A2SUAM1J3GNN3B", "asin": "0000013714", "overall": 5.0, "reviewText": "..."}  
{"reviewerID": "A2V48Q03FZUOSD", "asin": "0000013714", "overall": 5.0, "reviewText": "..."}

Attempting to load this with json.load() will trigger a JSONDecodeError immediately after the first line. The engineering pipeline must therefore utilize line-oriented parsers. The Python pandas library offers a read\_json(..., lines=True) method, but for large-scale ingestion (if scaling beyond the Beauty category), one must consider chunked reading or specialized engines like pyarrow to manage memory effectively.11

#### **2.2.3 Temporal and Metadata Considerations**

The version of the dataset typically used for these "small projects" is the 2014 or 2018 extraction. It is worth noting that a massive 2023 update exists, which includes 571 million reviews and fine-grained timestamps.13 However, the sheer size of the 2023 dataset (requiring massive sharding) makes it unsuitable for a local laptop-based project. The 2014 "Beauty" 5-core subset contains approximately 198,502 reviews, a "Goldilocks" size that fits in memory while remaining statistically significant.9

## ---

**3\. Pipeline Architecture and Reproducibility**

A recommender system is more than model code; it is a pipeline. To ensure this project is not merely a collection of scripts but a reproducible engineering artifact, we adopt the **Cookiecutter Data Science (CCDS)** standard.

### **3.1 Directory Structure and Philosophy**

The CCDS structure enforces a separation of concerns that is critical for ML projects.14

| Directory | Purpose | Recommender Context |
| :---- | :---- | :---- |
| data/raw | Immutable source data. | Stores ml-latest-small.zip and reviews\_Beauty\_5.json.gz. **Never edit these files.** |
| data/interim | Intermediate forms. | Unzipped CSVs, partially cleaned JSONL chunks. |
| data/processed | Modeling inputs. | ratings.parquet (optimized storage), tfidf\_features.npz (context vectors). |
| models/ | Serialized artifacts. | Pickled Surprise algo objects, Bandit policies. |
| src/ | Source code. | Scripts for downloading, parsing, and training. |
| notebooks/ | Exploration. | Jupyter notebooks for EDA and visualizing Regret/RMSE. |

This structure prevents the common error of "magic data"—datasets that exist only on one developer's machine in a modified state that cannot be recreated from the source.

### **3.2 Automation via Makefiles**

To operationalize this structure, we utilize a GNU Makefile. While often associated with C/C++ compilation, make is the ideal tool for defining the Directed Acyclic Graph (DAG) of a data pipeline.16

A Makefile defines **targets**, **dependencies**, and **recipes**. For our recommender project, the dependency graph is as follows:

1. **Target:** data/raw/ml-latest-small.zip  
   * **Dependency:** None (Source URL)  
   * **Recipe:** curl \-o $@ https://files.grouplens.org/...  
2. **Target:** data/processed/ratings.csv  
   * **Dependency:** data/raw/ml-latest-small.zip  
   * **Recipe:** unzip \-p $\< \> $@ (plus cleaning logic)  
3. **Target:** models/svd\_model.pkl  
   * **Dependency:** data/processed/ratings.csv  
   * **Recipe:** python src/models/train\_cf.py

By typing make models, the system automatically resolves the chain: it checks if the zip exists; if not, it downloads it. It checks if the CSV is extracted; if not, it extracts it. Then it runs the training. This idempotency is the hallmark of professional data engineering.17

### **3.3 Python Environment and Dependency Management**

A major finding in the research phase is the incompatibility of key recommender libraries with the latest Python versions.

#### **3.3.1 The scikit-surprise vs. Python 3.12 Conflict**

The scikit-surprise library, essential for Part 2 of the project, relies on the imp module for loading internal C-extensions. The imp module was deprecated in Python 3.4 and finally **removed** in Python 3.12.19 Furthermore, pre-compiled binary wheels for scikit-surprise are not widely available for Python 3.11/3.12 on all platforms (specifically Windows and some Linux distros), leading to compilation errors requiring Microsoft Visual C++ 14.0+ or gcc.20

**Implication:** The project **must** be constrained to Python 3.8, 3.9, or 3.10. Using Python 3.12 will cause the setup to fail catastrophically. The SETUP\_INSTRUCTIONS.md must explicitly enforce this constraint.

#### **3.3.2 Dependency Resolution**

The requirements.txt must act as the source of truth.

* **Core:** numpy, pandas, scipy.  
* **Modeling:** scikit-surprise (pinned to a version compatible with Py3.10), contextualbandits.  
* **Support:** scikit-learn (required by contextualbandits for base estimators), joblib.  
* **Ingestion:** pyarrow or fastparquet is recommended for handling the Amazon data if converted to columnar formats for speed.12

## ---

**4\. Part 1 Implementation: Data Ingestion Pipeline**

The first phase of the project involves establishing the "plumbing" that moves data from the internet to a trainable state.

### **4.1 MovieLens Ingestion Logic**

The ingestion of MovieLens data is straightforward but requires normalization.

1. **Download:** curl the zip file from the GroupLens permalink.  
2. **Extraction:** Unzip specifically the ratings.csv and movies.csv.  
3. **Normalization:** The standard ratings.csv contains a timestamp. For standard collaborative filtering in Surprise, this column acts as noise. The pipeline should include a preprocessing step (pandas script) that drops the timestamp and ensures the column order is \`\`—the format expected by the Surprise Reader.22

### **4.2 Amazon 5-Core Ingestion Logic**

The Amazon ingestion is more computationally intensive.

1. **Stream Decompression:** Do not unzip the .gz file to disk if storage is constrained. Python's gzip module can stream the file directly into pandas.  
2. **Parsing:** Use pandas.read\_json(filepath, lines=True).11  
3. **Transformation for Bandits:**  
   * **Binarization:** The contextualbandits library typically operates on binary rewards (Click/No-Click). The 5-star ratings must be converted: Ratings of 4 and 5 are mapped to 1 (positive reward); ratings 1-3 are mapped to 0\.6  
   * **Context Extraction:** The reviewText field is the source of the context. We must apply TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. This converts the unstructured text into a numerical vector $x\_t$ of size $V$ (vocabulary size). To keep dimensions manageable for LinUCB, we typically limit $V$ to 500–1000 features or apply Principal Component Analysis (PCA).23

## ---

**5\. Part 2 Implementation: Algorithms and Models**

This section details the theoretical and practical implementation of the two core modeling strategies.

### **5.1 Collaborative Filtering via Scikit-Surprise**

Collaborative Filtering (CF) operates on the "Guilt by Association" principle. If User A and User B rated items $X$ and $Y$ similarly, User A is likely to rate item $Z$ (which User B liked) similarly to User B.

#### **5.1.1 Latent Factor Models (SVD)**

The primary algorithm for this phase is Singular Value Decomposition (SVD), popularized during the Netflix Prize.  
Mathematically, we approximate the rating matrix $R$ as the product of two lower-rank matrices:

$$\\hat{r}\_{ui} \= \\mu \+ b\_u \+ b\_i \+ q\_i^T p\_u$$

Where:

* $\\mu$: Global average rating.  
* $b\_u, b\_i$: User and Item bias terms (e.g., a critical user who always rates low, or a popular movie that always gets 5 stars).  
* $q\_i, p\_u$: The latent feature vectors for item $i$ and user $u$.24

The optimization problem is to minimize the regularized squared error:

$$\\sum\_{r\_{ui} \\in R\_{train}} \\left(r\_{ui} \- \\hat{r}\_{ui} \\right)^2 \+ \\lambda \\left( b\_i^2 \+ b\_u^2 \+ ||q\_i||^2 \+ ||p\_u||^2 \\right)$$

This minimization is typically performed via Stochastic Gradient Descent (SGD) or Alternating Least Squares (ALS). Surprise uses SGD by default.8

#### **5.1.2 The Surprise Workflow**

1. **Reader Configuration:** We must instantiate a Reader object defining the rating scale (0.5, 5.0).22  
   Python  
   reader \= Reader(rating\_scale=(0.5, 5.0))  
   data \= Dataset.load\_from\_df(df\[\['userId', 'movieId', 'rating'\]\], reader)

2. **Cross-Validation:** To accurately report performance, we employ K-Fold cross-validation (typically $k=5$). The library splits the data, trains on $k-1$ folds, and tests on the remaining fold.25  
3. **Metrics:**  
   * **RMSE (Root Mean Squared Error):** Penalizes large errors heavily. Typical values for MovieLens Small are \~0.87.  
   * **MAE (Mean Absolute Error):** More interpretable average error margin. Typical values \~0.68.26

### **5.2 Contextual Bandits via contextualbandits**

While CF is powerful, it is static. Contextual Bandits allow for **Online Learning**, where the model updates itself after every interaction.

#### **5.2.1 The LinUCB Algorithm**

We will implement LinUCB (Linear Upper Confidence Bound). This algorithm assumes that the expected reward for an arm $a$ is a linear function of the context $x\_t$:

$$E\[r\_{t,a} | x\_t\] \= x\_t^T \\theta\_a$$

where $\\theta\_a$ is an unknown coefficient vector for arm $a$.  
The algorithm maintains an estimate $\\hat{\\theta}\_a$ and a confidence interval (uncertainty ellipsoid) around it. At each step $t$, it selects the arm that maximizes:

$$a\_t \= \\arg\\max\_{a \\in A} \\left( x\_t^T \\hat{\\theta}\_a \+ \\alpha \\sqrt{x\_t^T A\_a^{-1} x\_t} \\right)$$

The first term ($x\_t^T \\hat{\\theta}\_a$) represents exploitation (the current best estimate of reward). The second term ($\\alpha \\sqrt{\\dots}$) represents exploration (the uncertainty). As the algorithm gathers more data about an arm in a specific context, the uncertainty decreases, and the algorithm shifts from exploration to exploitation.6

#### **5.2.2 The "Simulated Bandit" (Replay) Methodology**

A major challenge in offline bandit research is that we only have "logged" data. We know what item the user *actually* bought and rated. We do not know what they *would have* done if our bandit algorithm had recommended a different item.

To solve this, we use **Rejection Sampling** (or the Replay Method) 23:

1. Stream through the historical log events $(x, a\_{log}, r\_{log})$.  
2. Let the bandit algorithm choose an action $\\pi(x)$.  
3. **If** $\\pi(x) \== a\_{log}$:  
   * The algorithms "agree." We reveal the reward $r\_{log}$ to the bandit.  
   * The bandit updates its parameters $\\theta$.  
   * We record this as a valid trial.  
4. **If** $\\pi(x) \\neq a\_{log}$:  
   * We cannot know the reward.  
   * We discard this event and move to the next.  
   * The bandit state remains unchanged.

This method is unbiased but data-inefficient, as many events are discarded. However, given the size of the Amazon dataset, enough matches typically occur to demonstrate learning convergence (decreasing Regret).

## ---

**6\. Implementation Setup Instructions**

The following content is designed to be saved as SETUP\_INSTRUCTIONS.md. It encapsulates the architectural decisions regarding directory structure, Python versioning, and command-line automation.

### **SETUP\_INSTRUCTIONS.md**

# **Recommender System Project: Setup & Execution Guide**

## **1\. Project Overview**

This project implements a dual-strategy recommendation engine:

1. **Collaborative Filtering:** Using the MovieLens (Small) dataset and Scikit-Surprise.  
2. **Contextual Bandits:** Using the Amazon Beauty (5-core) dataset and the ContextualBandits library.

## **2\. Critical Prerequisites**

### **2.1 Python Version Constraint**

**WARNING:** You must use **Python 3.8, 3.9, or 3.10**.

* **Do NOT use Python 3.12.** The scikit-surprise library relies on the imp module, which was removed in Python 3.12, causing build failures.  
* **Do NOT use Python 3.11** unless you are comfortable compiling C-extensions from source, as binary wheels are often missing.

### **2.2 System Dependencies**

* **C++ Compiler:** Required for installing contextualbandits and surprise.  
  * *Linux (Ubuntu/Debian):* sudo apt-get install build-essential  
  * *macOS:* Install Xcode Command Line Tools (xcode-select \--install).  
  * *Windows:* Install "Microsoft C++ Build Tools" (Visual Studio 2019 or later).

## **3\. Environment Installation**

It is highly recommended to use conda to manage the Python version and binary dependencies.

### **Option A: Conda (Recommended)bash**

# **Create environment with specific Python version**

conda create \-n recsys\_proj python=3.10  
conda activate recsys\_proj

# **Install base science stack**

conda install numpy pandas scipy scikit-learn

\#\#\# Option B: venv (Standard Python)  
\`\`\`bash  
\# Ensure you are running python 3.10  
python3.10 \-m venv venv  
source venv/bin/activate  \# Windows: venv\\Scripts\\activate

### **Install Project Libraries**

Run the following pip commands. Note that contextualbandits depends on scikit-learn.

Bash

pip install \--upgrade pip setuptools wheel  
pip install scikit-surprise  
pip install contextualbandits  
pip install notebook matplotlib pyarrow fastparquet

## **4\. Directory Structure (Cookiecutter Standard)**

Execute the following commands to establish the project hierarchy. This ensures separation of raw data and processed artifacts.

Bash

mkdir \-p data/raw          \# Immutable original zips/gz  
mkdir \-p data/interim      \# Unzipped/Intermediate files  
mkdir \-p data/processed    \# Cleaned CSVs/Parquets for models  
mkdir \-p models            \# Serialized model objects  
mkdir \-p notebooks         \# Jupyter notebooks  
mkdir \-p src/data          \# Scripts for downloading/parsing  
mkdir \-p src/models        \# Scripts for training

## **5\. Data Pipeline Execution**

We follow a strict "Raw to Processed" pipeline.

### **5.1 MovieLens Latest Small (Collaborative Filtering)**

* **Source:** GroupLens Research  
* **URL:** https://files.grouplens.org/datasets/movielens/ml-latest-small.zip

**Command to Download & Extract:**

Bash

\# Download  
curl \-o data/raw/ml-latest-small.zip \[https://files.grouplens.org/datasets/movielens/ml-latest-small.zip\](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)

\# Unzip (Junk paths to flatten structure into interim)  
unzip \-j data/raw/ml-latest-small.zip "ml-latest-small/ratings.csv" "ml-latest-small/movies.csv" \-d data/interim/

\# Validation  
\# data/interim/ratings.csv should exist.

### **5.2 Amazon Beauty 5-Core (Contextual Bandits)**

* **Source:** UCSD/McAuley Lab  
* **URL:** http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews\_Beauty\_5.json.gz

**Command to Download:**

Bash

\# Download  
curl \-o data/raw/reviews\_Beauty\_5.json.gz \[http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews\_Beauty\_5.json.gz\](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews\_Beauty\_5.json.gz)

\# Note: Do NOT unzip this file. The Python script will read the GZIP stream directly.

## **6\. Implementation Guidelines**

### **6.1 Part 1: Collaborative Filtering Script**

Create src/models/train\_cf.py.

* **Logic:**  
  1. Load data/interim/ratings.csv using Pandas.  
  2. Instantiate surprise.Reader(rating\_scale=(0.5, 5.0)).  
  3. Load data into surprise.Dataset.  
  4. Initialize surprise.SVD(n\_factors=100, n\_epochs=20, lr\_all=0.005, reg\_all=0.02).  
  5. Run surprise.model\_selection.cross\_validate(algo, data, measures=, cv=5, verbose=True).

### **6.2 Part 2: Contextual Bandits Script**

Create src/models/train\_bandit.py.

* **Logic:**  
  1. Stream data/raw/reviews\_Beauty\_5.json.gz.  
  2. **Context:** Use sklearn.feature\_extraction.text.TfidfVectorizer on the reviewText column. Limit features to 500 for speed (max\_features=500).  
  3. **Reward:** Create binary column: reward \= 1 if overall \>= 4, else 0\.  
  4. **Arms:** The dataset has thousands of items. To make the bandit converge in a reasonable time, filter the dataset to the top 10 or 20 most popular categories/items, or treat "Product Categories" as the arms.  
  5. **Simulation:** Use contextualbandits.online.BootstrappedUCB.  
     * Loop through the data.  
     * arm \= policy.predict(context\_vector)  
     * If arm \== actual\_item\_id: policy.partial\_fit(context\_vector, arm, reward).

\---

\#\# 7\. Comparative Analysis and Future Outlook

The implementation of these two parts reveals the fundamental trade-offs in recommender systems engineering.

\#\#\# 7.1 Performance vs. Adaptability  
The SVD model on MovieLens will likely achieve an RMSE of \~0.87. This is a highly accurate prediction of \*rating magnitude\*. However, it gives us no information about how to handle a new user who just joined the platform (Cold Start). The model requires a re-train (or at least a folding-in process) to generate embeddings for the new user.

In contrast, the LinUCB bandit on Amazon data is explicitly designed for the cold start. By leveraging the TF-IDF vectors of the reviews (or potentially the user's initial search query in a production app), the bandit can make an informed recommendation ($x\_t^T \\theta\_a$) immediately. However, the "Regret" metric typically starts high and decays over time, implying that early users are "sacrificed" to gather information that benefits later users.

\#\#\# 7.2 Scalability and Real-World Constraints  
While this project uses "Small" and "5-core" datasets, scaling to the full MovieLens 25M or the Amazon 2023 dataset (571M reviews) requires a fundamental architectural shift.  
\*   \*\*Memory:\*\* The \`Surprise\` library loads the full interaction matrix into RAM. For 25M ratings, this is feasible on a large server but risky. For 571M interactions, one must switch to \*\*Spark ALS\*\* (distributed matrix factorization) or deep learning approaches (Neural Collaborative Filtering) that train via mini-batches.\[28, 29\]  
\*   \*\*Latency:\*\* The LinUCB algorithm involves matrix inversion ($A^{-1}$) which is $O(d^3)$ complexity where $d$ is the number of context features. If using BERT embeddings ($d=768$), real-time inversion becomes a latency bottleneck. Production systems often use \*\*Thompson Sampling\*\* (which avoids inversion via sampling) or approximations like \*\*Linear epsilon-Greedy\*\* to maintain sub-millisecond response times.

\#\#\# 7.3 Conclusion  
This project demonstrates that building a recommender system is an exercise in constraint management. We constrain the dataset (5-core) to ensure learnability. We constrain the Python environment (3.10) to ensure library stability. We constrain the directory structure (CCDS) to ensure reproducibility. By mastering these constraints across both static (CF) and dynamic (Bandit) paradigms, one gains the holistic engineering perspective necessary for deploying modern personalized systems.

\---

\#\# References in Context (Integrated Analysis)

Throughout this report, we have synthesized information from various technical sources to construct a robust pipeline.  
\*   \*\*Dataset Definitions:\*\* The characterization of MovieLens follows the official GroupLens specifications \[2, 7\], while the Amazon 5-core structure is derived from the McAuley Lab's documentation.\[9, 10\]  
\*   \*\*Library Constraints:\*\* The critical Python 3.12 warning is based on recent community discussions regarding \`scikit-surprise\` build failures.\[19, 20, 21\]  
\*   \*\*Algorithmic Theory:\*\* The bandit implementation details rely on the documentation of the \`contextualbandits\` library and its implementation of LinUCB/Thompson Sampling.\[6, 23, 30\]  
\*   \*\*Engineering Standards:\*\* The directory structure and automation recommendations are aligned with the Cookiecutter Data Science philosophy  and Makefile best practices.

#### **Works cited**

1. MovieLens Dataset: The Essential Benchmark for Recommender Systems | Shaped Blog, accessed January 11, 2026, [https://www.shaped.ai/blog/movielens-dataset-the-essential-benchmark-for-recommender-systems](https://www.shaped.ai/blog/movielens-dataset-the-essential-benchmark-for-recommender-systems)  
2. MovieLens Latest Datasets | GroupLens, accessed January 11, 2026, [https://grouplens.org/datasets/movielens/latest/](https://grouplens.org/datasets/movielens/latest/)  
3. Movielens Latest Dataset \- Kaggle, accessed January 11, 2026, [https://www.kaggle.com/datasets/deepak1011/movielens-latest-datasets](https://www.kaggle.com/datasets/deepak1011/movielens-latest-datasets)  
4. Ultimate Guide to Contextual Bandits: From Theory to Python Implementation, accessed January 11, 2026, [https://findingtheta.com/blog/ultimate-guide-to-contextual-bandits-from-theory-to-python-implementation](https://findingtheta.com/blog/ultimate-guide-to-contextual-bandits-from-theory-to-python-implementation)  
5. Amazon Reviews'23, accessed January 11, 2026, [https://amazon-reviews-2023.github.io/](https://amazon-reviews-2023.github.io/)  
6. david-cortes/contextualbandits: Python implementations of contextual bandits algorithms \- GitHub, accessed January 11, 2026, [https://github.com/david-cortes/contextualbandits](https://github.com/david-cortes/contextualbandits)  
7. movielens | TensorFlow Datasets, accessed January 11, 2026, [https://www.tensorflow.org/datasets/catalog/movielens](https://www.tensorflow.org/datasets/catalog/movielens)  
8. Surprise · A Python scikit for recommender systems., accessed January 11, 2026, [https://surpriselib.com/](https://surpriselib.com/)  
9. Amazon review data, accessed January 11, 2026, [https://cseweb.ucsd.edu/\~jmcauley/datasets/amazon/links.html](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)  
10. Amazon review data, accessed January 11, 2026, [https://jmcauley.ucsd.edu/data/amazon/](https://jmcauley.ucsd.edu/data/amazon/)  
11. pandas.read\_json — pandas 2.3.3 documentation \- PyData |, accessed January 11, 2026, [https://pandas.pydata.org/docs/reference/api/pandas.read\_json.html](https://pandas.pydata.org/docs/reference/api/pandas.read_json.html)  
12. JSON Lines Reading with pandas 100x Faster Using NVIDIA cuDF | NVIDIA Technical Blog, accessed January 11, 2026, [https://developer.nvidia.com/blog/json-lines-reading-with-pandas-100x-faster-using-nvidia-cudf/](https://developer.nvidia.com/blog/json-lines-reading-with-pandas-100x-faster-using-nvidia-cudf/)  
13. McAuley-Lab/Amazon-Reviews-2023 · Datasets at Hugging Face, accessed January 11, 2026, [https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)  
14. Cookiecutter Data Science: A Standardized, Flexible Approach for Modern Data Projects, accessed January 11, 2026, [https://pub.towardsai.net/cookiecutter-data-science-a-standardized-flexible-approach-for-modern-data-projects-84923e1fb890](https://pub.towardsai.net/cookiecutter-data-science-a-standardized-flexible-approach-for-modern-data-projects-84923e1fb890)  
15. Cookiecutter Data Science, accessed January 11, 2026, [https://cookiecutter-data-science.drivendata.org/](https://cookiecutter-data-science.drivendata.org/)  
16. The Case for Makefiles in Python Projects (And How to Get Started) \- KDnuggets, accessed January 11, 2026, [https://www.kdnuggets.com/the-case-for-makefiles-in-python-projects-and-how-to-get-started](https://www.kdnuggets.com/the-case-for-makefiles-in-python-projects-and-how-to-get-started)  
17. Make for Data Science, accessed January 11, 2026, [https://datasciencesouth.com/blog/make/](https://datasciencesouth.com/blog/make/)  
18. Makefile Tutorial | Towards Data Science, accessed January 11, 2026, [https://towardsdatascience.com/a-data-scientists-guide-to-make-and-makefiles-1595f39e0704/](https://towardsdatascience.com/a-data-scientists-guide-to-make-and-makefiles-1595f39e0704/)  
19. Getting requirements to build wheel ... error, accessed January 11, 2026, [https://discuss.python.org/t/getting-requirements-to-build-wheel-error/39906](https://discuss.python.org/t/getting-requirements-to-build-wheel-error/39906)  
20. Issues with installing Wheel \- Python Discussions, accessed January 11, 2026, [https://discuss.python.org/t/issues-with-installing-wheel/42302](https://discuss.python.org/t/issues-with-installing-wheel/42302)  
21. scikit-surprise pip installation with multiple errors/notes (e.g. errors from subprocess, absent from the \`packages\` configuration) \- Stack Overflow, accessed January 11, 2026, [https://stackoverflow.com/questions/77789068/scikit-surprise-pip-installation-with-multiple-errors-notes-e-g-errors-from-su](https://stackoverflow.com/questions/77789068/scikit-surprise-pip-installation-with-multiple-errors-notes-e-g-errors-from-su)  
22. Reader class \- Surprise' documentation\! \- Read the Docs, accessed January 11, 2026, [https://surprise.readthedocs.io/en/stable/reader.html](https://surprise.readthedocs.io/en/stable/reader.html)  
23. contextualbandits/example/online\_contextual\_bandits.ipynb at master \- GitHub, accessed January 11, 2026, [https://github.com/david-cortes/contextualbandits/blob/master/example/online\_contextual\_bandits.ipynb](https://github.com/david-cortes/contextualbandits/blob/master/example/online_contextual_bandits.ipynb)  
24. Collaborative Filtering Libraries in Python \- Medium, accessed January 11, 2026, [https://medium.com/top-python-libraries/collaborative-filtering-libraries-in-python-2f1cf45801b7](https://medium.com/top-python-libraries/collaborative-filtering-libraries-in-python-2f1cf45801b7)  
25. Getting Started — Surprise 1 documentation, accessed January 11, 2026, [https://surprise.readthedocs.io/en/stable/getting\_started.html](https://surprise.readthedocs.io/en/stable/getting_started.html)  
26. scikit-surprise \- PyPI, accessed January 11, 2026, [https://pypi.org/project/scikit-surprise/](https://pypi.org/project/scikit-surprise/)  
27. Scalable and Interpretable Contextual Bandits: A Literature Review and Retail Offer Prototype \- arXiv, accessed January 11, 2026, [https://arxiv.org/html/2505.16918v1](https://arxiv.org/html/2505.16918v1)