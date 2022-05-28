{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "vsYlmmTz7d2T"
      },
      "source": [
        "#***Neural Network for Recommendation Systems***\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "0-mfshQL-Gw_"
      },
      "source": [
        "<img src=\"images/IMG_1091.jpg\" alt=\"drawing\" width=\"500\"/>\n",
        "\n",
        "So when we talk about Recomender System in the real-world we have to imagine an\n",
        "hybrid system, in whitch Content-Based, Collaborative Filtering and knowledge-based are implement in the same model.\n",
        "\n",
        "<img src=\"images/IMG_1096.jpg\" alt=\"drawing\" width=\"500\"/>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "PLzJi_be-Xl1"
      },
      "source": [
        "##***Structured and Unstructured Data***\n",
        "\n",
        "***Content-based recommendation models***\n",
        "\n",
        "<img src=\"images/IMG_1093.jpg\" alt=\"drawing\" width=\"500\"/>\n",
        "\n",
        "***Collaborative filtering***\n",
        "\n",
        "<img src=\"images/IMG_1094.jpg\" alt=\"drawing\" width=\"500\"/>\n",
        "\n",
        "\n",
        "***Knowledge-based***\n",
        "\n",
        "<img src=\"images/IMG_1095.jpg\" alt=\"drawing\" width=\"500\"/>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "1l-KMLwGrMk_"
      },
      "source": [
        "***Hybrid model***\n",
        "\n",
        "<img src=\"images/IMG_1092.jpg\" alt=\"drawing\" width=\"500\"/>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Ce6Xem5nXdiN"
      },
      "source": [
        "##***Content-Based Filtering overview***\n",
        "Uses item features to recommend new items that are similar to what the user has liked in the past. Content-based does not take into account the behavior or ratings of other users.\n",
        "\n",
        "<img src=\"images/IMG_1119.jpg.jpg\" alt=\"drawing\" width=\"500\"/>\n",
        "\n",
        "<img src=\"images/IMG_1118.jpg.jpg\" alt=\"drawing\" width=\"500\"/>\n",
        "\n",
        "The user space and product space are ***sparse*** and ***skewed***. This means that most item are rated by very few users, and most users rate only a small fraction of items. Furthermore, some properties are very popular and some users are very prolific."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "8ROdoQAmYQjb"
      },
      "source": [
        "To feed a Neural Networks for Recommendetion System, cause of the sparse dataset we define the features columns to use in our model into a dense representation. TensorFlow allows us to do this using [various feature columns](https://www.tensorflow.org/api_docs/python/tf/feature_column).\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "efRb9sfNYO-8"
      },
      "source": [
        "***Examples of feature columns:***\n",
        "\n",
        "- [`tf.feature_column.categorical_column_with_hash_bucket\n",
        "`](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_hash_bucket)\n",
        "- [`tf.feature_column.embedding_column`](https://www.tensorflow.org/api_docs/python/tf/feature_column/embedding_column)\n",
        "\n",
        "\n",
        "    items_id_column = tf.feature_column.categorical_column_with_hash_bucket(\n",
        "    key=\"items_id\",\n",
        "    hash_bucket_size= len(content_ids_list) + 1)\n",
        "\n",
        "    embedded_content_column = tf.feature_column.embedding_column(\n",
        "    categorical_column=items_id_column,\n",
        "    dimension=10)\n",
        "\n",
        "- [`tf.feature_column.categorical_column_with_vocabulary_list\n",
        "`](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_vocabulary_list)\n",
        "- [`tf.feature_column.indicator_column`](https://www.tensorflow.org/api_docs/python/tf/feature_column/indicator_column)\n",
        "\n",
        "\n",
        "    category_column_categorical = tf.feature_column.categorical_column_with_vocabulary_list(\n",
        "    key=\"category\",\n",
        "    vocabulary_list=categories_list,\n",
        "    num_oov_buckets=1)\n",
        "\n",
        "    category_column = tf.feature_column.indicator_column(category_column_categorical)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gGg2KqG4uh6Q"
      },
      "source": [
        "We use the input_layer feature column to create the dense input layer to our network. We can adjust the number of hidden units as a parameter.\n",
        "\n",
        "    net = tf.feature_column.input_layer(features, params['feature_columns'])\n",
        "    for units in params['hidden_units']:\n",
        "        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)\n",
        "     #Compute logits (1 per class).\n",
        "    logits = tf.layers.dense(net, params['n_classes'], activation=None)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "rI1TqyEgXFLg"
      },
      "source": [
        "##***Collaborative Filtering overview***\n",
        "Collaborative filtering recommendations use similarities between items and the user simultaneously in an embedding space. In collaborative filtering the ***only thing we need are: user_id, item-id and rating***. The latter can be ***explicit*** or ***implicit***. Another importan thing is to trasfom all the variables in integer.\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "zBAIxEdJtQnO"
      },
      "source": [
        "***Embedding space***\n",
        "\n",
        "Each user and item is a k-dimensional point within an embedding space. Embeddings can be ***learned from data***. The idea is to compressing the data to find the best generalites to rely on, called ***latent factors***. The factorization split the users-interactions matrix into ***row factors*** and ***column factors*** that are essentially user and item embedding."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "0v-aQfkJv9Dm"
      },
      "source": [
        "Our goal is to factorize the ratings matrix $A$ into the product of a user embedding matrix $U$ and movie embedding matrix $V$, such that:\n",
        "\n",
        " $A \\approx UV^\\top$ with\n",
        "$U = \\begin{bmatrix} u_{1} \\\\ \\hline \\vdots \\\\ \\hline u_{N} \\end{bmatrix}$ and\n",
        "$V = \\begin{bmatrix} v_{1} \\\\ \\hline \\vdots \\\\ \\hline v_{M} \\end{bmatrix}$.\n",
        "\n",
        "- $N$ is the number of users,\n",
        "- $M$ is the number of movies,\n",
        "- $A_{ij}$ is the rating of the $j$th movies by the $i$th user,\n",
        "- each row $U_i$ is a $d$-dimensional vector (embedding) representing user $i$,\n",
        "- each rwo $V_j$ is a $d$-dimensional vector (embedding) representing movie $j$,\n",
        "- the prediction of the model for the $(i, j)$ pair is the dot product $\\langle U_i, V_j \\rangle$.\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "p62y8SR0wmhn"
      },
      "source": [
        "***Sparse Representation of the Rating Matrix***\n",
        "\n",
        "As we have already mention the rating matrix could be very sparse. In our help comes [tf.SparseTensor](https://www.tensorflow.org/api_docs/python/tf/SparseTensor). \n",
        "\n",
        "A `SparseTensor` uses three tensors to represent the matrix: `tf.SparseTensor(indices, values, dense_shape)` represents a tensor, where a value $A_{ij} = a$ is encoded by setting `indices[k] = [i, j]` and `values[k] = a`. The last tensor `dense_shape` is used to specify the shape of the full underlying matrix.\n",
        "\n",
        "***Example***\n",
        "\n",
        "Assume we have $2$ users, $4$ movies and $3$ ratings:\n",
        "\n",
        "user\\_id | movie\\_id | rating\n",
        "--:|--:|--:\n",
        "0 | 0 | 5.0\n",
        "0 | 1 | 3.0\n",
        "1 | 3 | 1.0\n",
        "\n",
        "The corresponding rating matrix is:\n",
        "\n",
        "$$\n",
        "A =\n",
        "\\begin{bmatrix}\n",
        "5.0 & 3.0 & 0 & 0 \\\\\n",
        "0   &   0 & 0 & 1.0\n",
        "\\end{bmatrix}\n",
        "$$\n",
        "\n",
        "And the SparseTensor representation is:\n",
        "\n",
        "    SparseTensor(\n",
        "    indices=[[0, 0], [0, 1], [1,3]],\n",
        "    values=[5.0, 3.0, 1.0],\n",
        "    dense_shape=[2, 4])\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "dbYMra_1zutp"
      },
      "source": [
        "***Weighted ALS (WALS)***\n",
        "\n",
        "There are many ways to handle unobserved user-interaction matrix pair. ***SVD*** explicitly sets all missing values to zero. ***ALS*** simply ingnores missing values. ***WALS*** uses weights instead of zeros that can be thought of a representing ***low confidence***.\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "h8kPr_qfu_A1"
      },
      "source": [
        "***WALS***\n",
        "\n",
        "\\begin{eqnarray}\n",
        "        \\sum_{(i,j)\\in obs}(A_{i,j}-U_iV_j)^2+w_0 \\sum_{(i,j)\\notin obs}(0-U_iV_j)^2\n",
        "    \\end{eqnarray}\n",
        "\n",
        "An existing `WALSMatrixFactorization`\n",
        "function in TensorFlow `tf.__version__ = 1.15` can be used to do ***matrix factorization with*** ***WALS***.\n",
        "\n",
        "The main idea is the following:\n",
        "\n",
        "Iterative:\n",
        "- $U$ and $V$ are randomly generated,\n",
        "- Fix $U$ and find, by solving a linear system, the best $V$,\n",
        "- Fix $V$ and find, by solving a linear system, the best $U$.\n",
        "\n",
        "The algorithm is guaranteed to converge and can be parallelised.\n",
        "\n",
        "<img src=\"images/IMG_1123.jpg\" alt=\"drawing\" width=\"500\"/>\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "RlXL6aq63yqr"
      },
      "source": [
        "The code for ***WALS using TensorFlow*** can be found at the following link [].\n",
        "Below it is show how works remapping keys to SparseTensor to fix re-indexing after batching.\n",
        "\n",
        "<img src=\"images/IMG_1113.jpg\" alt=\"drawing\" width=\"500\"/>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "8JLYV6GJss1j"
      },
      "source": [
        "##***Context-aware recommendation system CARS***\n",
        "An important aspect concerns the context in whitch the user perceives the experience. We are talk about to Adding Context.\n",
        "\n",
        "- An item is not just an item.\n",
        "- A user is not just a user.\n",
        "- The context it is experienced in changes perception.\n",
        "- This affects sentiment.\n",
        " "
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "4RrxllMluQ3g"
      },
      "source": [
        "There are three main types of ***CARS*** algorithms:\n",
        "- ***Contextual prefiltering,***\n",
        "- ***Contextual postfiltering,***\n",
        "- ***Contextual modeling.***\n",
        "     "
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "KrfyIR1Vuxk0"
      },
      "source": [
        "###***Contextual prefiltering***\n",
        "\n",
        "<img src=\"images/IMG_1097.jpg\" alt=\"drawing\" width=\"500\"/>\n",
        "\n",
        "<img src=\"images/IMG_1098.jpg\" alt=\"drawing\" width=\"500\"/>\n",
        "\n",
        "<img src=\"images/IMG_1099.jpg\" alt=\"drawing\" width=\"500\"/>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "NzO4mNoU1gIu"
      },
      "source": [
        "This is easy to see when we have a small toy data set like this,\n",
        "but how would we do it for much larger and more complex data sets?\n",
        "We simply can use a ***t-test*** on two chunks of ratings,\n",
        "and ***choose what gives the maximum t value,\n",
        "and thus the smallest p-value***.\n",
        "There's a simple splitting,\n",
        "when splitting across one dimension of context,\n",
        "and complex splitting was putting over multiple dimensions of contexts.\n",
        "***Complex splitting can have sparsity issues and can have overfitting problems***.\n",
        "So, single splitting is often used to avoid these issues. \n",
        "There's also user splitting which it's extremely similar to item splitting,\n",
        "except now we split along user rather than item"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "B-N7T31qwQnp"
      },
      "source": [
        "###***Contextual postfiltering***\n",
        "\n",
        "<img src=\"images/IMG_1100.jpg\" alt=\"drawing\" width=\"500\"/>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Vb5BkTtj4A-h"
      },
      "source": [
        "What happens to all the context dimensions?\n",
        "Well, we simply ignore them.\n",
        "We ignore a context.\n",
        "We process the data as if it was just a user item interaction matrix. \n",
        "We then apply our users vector to\n",
        "the output embeddings to get the representation in embedding space.\n",
        "This gives us our recommendations.\n",
        "But these are exactly the same as if we never had\n",
        "contexts data. how do we fix this problem?\n",
        "Well, we can try to adjust\n",
        "our non-contractual recommendations by applying the context back in.\n",
        "\n",
        "For example, if our user from before still wants to see a movie on\n",
        "Thursday after work and on Thursday they usually watch action movies,\n",
        "our postfiltering can filter out all non-action movies\n",
        "for the recommendations returned by our non-contractual recommender.\n",
        "This then gets us finally to the contextual recommendations that we wanted. "
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ZnJRqfPJwY45"
      },
      "source": [
        "###***Contextual modeling***\n",
        "\n",
        "<img src=\"images/IMG_1101.jpg\" alt=\"drawing\" width=\"500\"/>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "w9qqCqxf5K4I"
      },
      "source": [
        "***Deviation-Based Context-Aware Matrix Factorization (2011)***\n",
        "\n",
        "\n",
        "In deviation-base context-aware matrix factorization,\n",
        "we want to know how a user's rating is deviated across contexts.\n",
        "This difference is called the ***contextual rating deviation***, or ***CRD***.\n",
        "It looks at the deviations of users across contexts dimensions"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "14cqaL0-84hW"
      },
      "source": [
        "<img src=\"images/IMG_1102.jpg\" alt=\"drawing\" width=\"500\"/>\n",
        "\n",
        "<img src=\"images/IMG_1103.jpg\" alt=\"drawing\" width=\"500\"/>\n",
        "\n",
        "***Traditional recommendation system***: use standard matrix factorization or bias matrix factorization.\n",
        "Where we add a term for the global average rating,\n",
        "a bias term for user u,\n",
        "and a bias term for item i.\n",
        "Of course, we have our user item interaction term,\n",
        "which is the dot product of the users vector p,\n",
        "from the user factor embedding matrix u,\n",
        "and the items factor vector q,\n",
        "from the item factor embedding matrix v.\n",
        "\n",
        "\\begin{eqnarray}\n",
        "        r^*_{ui} = \\mu + b_u + b_i + p_u^Tq_i.\n",
        "    \\end{eqnarray}\n",
        "\n",
        "***Context-aware matrix factorization***: We can see, that almost everything is the same,\n",
        "except for two terms.\n",
        "On the right-hand side,\n",
        "we have added the contextual rating deviations, summed across contexts.\n",
        "This gives us contextual multidimensional ratings on the left-hand side.\n",
        "\n",
        "\\begin{eqnarray}\n",
        "        r^*_{uic_1c_2...c_N} = \\mu + b_u + b_i + p_u^Tq_i + \\sum_{j=1}^CRD(c_j).\n",
        "    \\end{eqnarray}\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "n-J5LeA1_1NV"
      },
      "source": [
        "##***YouTube Recommendation System Case Study***\n",
        "\n",
        "YouTube's Recommender System consists of\n",
        "two neural networks and many different data sources.\n",
        "There's a candidate generation network that accepts millions of video corpuses.\n",
        "The output of this network, is the input to a ranking network combined with other candidate sources,\n",
        "which could be things like videos in the news for freshness,\n",
        "videos for neighboring videos,related videos, or sponsored videos.\n",
        "This is also combined with video features."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "nwxoxFyWIag-"
      },
      "source": [
        "<img src=\"images/IMG_1104.jpg\" alt=\"drawing\" width=\"500\"/>\n",
        "\n",
        "***High precision*** means that every candidate generated is relevant to user.\n",
        "Obviously, this is important because users do not want to be shown\n",
        "irrelevant videos.\n",
        "\n",
        "***High recall*** means that it will\n",
        "recommend things the user will definitely like."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "JPkuOTvHIadj"
      },
      "source": [
        "###***Candidate generation***\n",
        "\n",
        "<img src=\"images/IMG_1087.jpg\" alt=\"drawing\" width=\"500\"/>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "p7qhsek2Q-T7"
      },
      "source": [
        "The first step for candidate generation is see the item embeddings. We find the last 10 videos watched by the user and\n",
        "use the embeddings to get their vectors within embedding space. Than we average the embeddings of those 10 videos, so we'll have a resultant single embedding that is the average along each embedding dimension. This becomes the watch vector,\n",
        "which will be one of the features to our deep neural network.\n",
        "\n",
        "We would do the same thing with past search queries.\n",
        "Collaborative filtering for \n",
        "next search term is a collaborative filtering model that is similar to \n",
        "the user history based collaborative filter we talked about earlier.\n",
        "Essentially, this is like doing to word to vec on pairs of search terms.\n",
        "We will find an average search and this will become our search vector.\n",
        "Another input feature to our deep neural network.\n",
        "\n",
        "We also should add any knowledge we have about the user.\n",
        "Location is important so you just conceive localised videos and\n",
        "also because of language."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "xAHa7K1CV4zL"
      },
      "source": [
        "<img src=\"images/IMG_1088.jpg\" alt=\"drawing\" width=\"500\"/>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "B-1zt8NdVtlr"
      },
      "source": [
        "Taking the top n of these videos is probably what we want.\n",
        "These come out of a ***softmax*** for training. \n",
        "There's also a benefit to finding the closest users and generating those candidates as well (*this is the way that viral videos are created*).\n",
        "Therefore, we can treat the layer right before softmax as a user embedding.\n",
        "This is also a benefit to finding videos related content wise to the video that are currently watching.\n",
        "Therefore, we can use the output of the DNN Classifier as video vectors.\n",
        "This compounded with the user embeddings layer generates candidates during serving, so nearest neighbors consist of neighboring users and neighboring videos.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "FxUL1D8TIwBB"
      },
      "source": [
        "###***Ranking***\n",
        "The first step, is to take the video suggest to the user.\n",
        "These get combined with the videos watched by the user, both get used as individual video embeddings, and as an average embedding. These input features all feed through the ranking neural networks layers, the output of the DNN classifier is the ranking.\n",
        "\n",
        "<img src=\"images/IMG_1089.jpg\" alt=\"drawing\" width=\"500\"/>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "kIgPUR7WbPMq"
      },
      "source": [
        "During training time, the output is a weighted logistic function, \n",
        "whereas during serving, it as just a logistic. \n",
        "Serving uses logistic regression for scoring the videos and is optimized for expected watch time from the training labels. \n",
        "This is used instead of expected click, \n",
        "because then the recommendations could favor clickbait videos over actual good recommendations.\n",
        "For training because we are using the expected watch time, we use the weighted logistic instead.\n",
        "The watch time is used as the weight for positive interactions, and negative interactions just get a unit weight.\n",
        "\n",
        "<img src=\"images/IMG_1090.jpg\" alt=\"drawing\" width=\"500\"/>\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Iu8jGZnMACOt"
      },
      "source": [
        "#***Example: Softmax DNN for Recommendation (Movies Recommendation)***\n",
        "Softmax treats the problem as a multiclass prediction problem in which:\n",
        "\n",
        "- The input is the user query.\n",
        "- The output is a probability vector with size equal to the number of items in the corpus, representing the probability to interact with each item; for example, the probability to click on or watch a YouTube video.\n",
        "  "
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "qsHQMF9Me_vZ"
      },
      "source": [
        "***Input***\n",
        "\n",
        "- ***dense features*** (for xample, watch time and time since last watch).\n",
        "-***Sparse features***(for example, watch history and country).\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "LBMpGeJQfuI-"
      },
      "source": [
        "***Model***\n",
        "\n",
        "The model is based adding hidden layers whit non-linear activation functions (i.e.ReLu, Leaky ReLu ...). This speed up the reserch of the minimum during the optimization, and capture more complex relatioships in the data. We will denote the output of the last hidden layer by $ψ(x) \\in ℜ^d$.\n",
        "\n",
        "The model maps the output of the last hidden layer $ψ(x)$ throught a softmax layer to a probability distribution $P=h(\\psi(x)V^T)$, where:\n",
        "- $h: ℜ^m \\to ℜ^m$, is the softmax function.\n",
        "- $V \\in \\Re^{mxd}$ is the matrix of weights of the softmax layer.\n",
        "   \n",
        "<img src=\"images/softmaxmodel.png\" alt=\"drawing\" width=\"500\"/>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "vHpUw3zzpIZj"
      },
      "source": [
        "***Softmax cross entropy loss***\n",
        "Entropy of a random variable X is the level of uncertainty inherent in the variables possible outcome.\n",
        "\n",
        "Is the generalization of the classical ***binary cross-entropy***. Recall that the softmax model maps the input features x to a user embedding $ψ(x)∈R^d$, where $d$ is the embedding dimension. This vector is then multiplied by a movie embedding matrix $V∈R^{m×d}$ (where $m$ is the number of movies), and the final output of the model is the softmax of the product.\n",
        "\n",
        "\\begin{eqnarray}\n",
        "        p^* = softmax(\\psi(x)V^T).\n",
        "    \\end{eqnarray}\n",
        "\n",
        "Given a target label $y$, if we denote by $p=1_y$ a one-hot encoding of this target label, then the loss is the ***cross-entropy*** between $p^*(x)$ and $p$.\n",
        "\n",
        "Using TensorFlow, we will write a function that takes tensors representing the user embeddings $\\psi(x)$, movie embeddings $V$, target label $y$, and return the cross-entropy loss, using the function [`tf.nn.sparse_softmax_cross_entropy_with_logits`](https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits), which takes `logits` as input, where `logits` refers to the product $\\psi(x) V^\\top$."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "cfXs3G1OmXJ8"
      },
      "source": [
        "***Negative Sampling and Folding***\n",
        "\n",
        "Since the loss function compares two probability vectors $p, p(x) \\in \\Re^n$ (the ground truth and the output of the model, respectively), computing the gradient of the loss (for a single query) can be prohibitively expensive if the corpus size is too big.\n",
        "\n",
        "You could set up a system to compute gradients only on the ***positive items (items that are active in the ground truth vector)***. However, if the system only trains on positive pairs, the model may suffer from ***folding***.\n",
        "\n",
        "<img src=\"images/folding.png\" alt=\"drawing\" width=\"500\"/>\n",
        "\n",
        "In the following figure, assume that each color represents a different category of queries and items. Each query (represented as a square) only mostly interacts with the items (represented as a circle) of the same color (different language in YouTube9.\n",
        "\n",
        "The model may learn how to place the query/item embeddings of a given color relative to each other (correctly capturing similarity within that color), but embeddings from different colors may end up in the same region of the embedding space, by chance. This phenomenon, known as ***folding***, can lead to spurious recommendations: at query time, ***the model may incorrectly predict a high score for an item from a different group***.\n",
        "\n",
        "***Negative examples*** are items labeled \"irrelevant\" to a given query. Showing the model negative examples during training teaches the model that embeddings of different groups should be pushed away from each other.\n",
        "\n",
        "More precisely, we compute an approximate gradient, using the following items:\n",
        "\n",
        "- All positive items (the ones that appear in the target label)\n",
        "- A sample of negative items ($j$ in $1,2,...,n)$\n",
        "\n",
        "One possible strategies for sampling negatives is given higher probability to items $j$ with higher score $ψ(x)$. these are examples that contribute the most to the gradient); these examples are often called ***hard negatives***. "
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "2mYQiI91lcAT"
      },
      "source": [
        "***DNN and Matrix Factorization***\n",
        "\n",
        "In both the softmax model and the matrix factorization model, the system learns one embedding vector $V_j$ per item $j$. What we called the item embedding matrix $V \\in ℜ^{nxd}$ in matrix factorization is now the matrix of weights of the softmax layer.\n",
        "\n",
        "The query embeddings, however, are different. Instead of learning one embedding $U_i$ per query $i$, the system learns a mapping from the query feature $x$ to an embedding $ψ(x) \\in \\Re^n$. Therefore, you can think of this DNN model as a generalization of matrix factorization, in which you replace the query side by a nonlinear function $\\psi(x)$."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "mkgqsmkvAekj"
      },
      "source": [
        "#***References***\n",
        "[`Deep Neural Networks for YouTube Recommendations, Paul Covington, Jay Adams, Emre Sargin`](https://research.google/pubs/pub45530/)\n",
        "\n",
        "[`Recommendation Systems with TensorFlow on GCP`](https://www.coursera.org/learn/recommendation-models-gcp)\n",
        "\n",
        "[`Collaborative Filtering for Implicit Feedback Datasets, Hu, Koren, Volinsky`](http://yifanhu.net/PUB/cf.pdf)\n",
        "\n",
        "[`Altair: Declarative Visualization in Python`](https://altair-viz.github.io/)\n",
        "\n",
        "[`Recommended Reading for IR Research Students`](http://delab.csd.auth.gr/~dimitris/courses/ir_spring07/papers/Recommended%20reading%20for%20IR%20research%20students.pdf)\n"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "collapsed_sections": [
        "ZwKXq5ypnucm",
        "DqxfOZkK1ToU",
        "vxG9qeLPrAyJ",
        "eifa-J9Hr314",
        "d9AeRtOus0iq",
        "FSnDJ22w3mIj",
        "pGcmbzfF4ryV",
        "0Wv2bAxl6vFH"
      ],
      "name": "Recommendation_system_DNN.ipynb",
      "provenance": []
    },
    "environment": {
      "kernel": "python3",
      "name": "tf-gpu.1-15.m87",
      "type": "gcloud",
      "uri": "gcr.io/deeplearning-platform-release/tf-gpu.1-15:m87"
    },
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.7.12"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
