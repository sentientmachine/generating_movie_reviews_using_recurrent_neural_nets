
# Generating movie reviews using recurrent neural nets:

The purpose of this project is to demonstrate my coding and writing skills on the subject of Tensorflow, Keras RNN's, Python, matplotlib, and numpy to create an endless supply of machine-generated positive and negative natural-English paragraph movie reviews.


# Training Dataset Download source:

The data to train the recurrent neural network comes from anonymous English movie reviews from Stanford's "Large movie review" dataset in traditional written English paragraph form.

80MB download the `aclImdb` tar file from: http://ai.stanford.edu/~amaas/data/sentiment

# Parse the dataset to be useful for our purposes: 

Extract the `aclImdb` tar file locally, produces a directory.

The subset of the data I want is in: `~/aclImdb/train/pos` and `~/aclImdb/train/neg`.

Each positive and negative review is in its own text file.  So data preparation and cleaning is required.

# Data conversion, transformation, cleaning, preparation:

A short script can join all the separate review files into a single file that has a different positive reivew on every line.  And also the same for all negative reviews.

Keeping things as simple as possible, bash commands can do this quickly:

    cd ~/aclImdb/train/pos
    cat * > positive_movie_reviews.txt

And

    cd ~/aclImdb/train/neg
    cat * > negative_movie_reviews.txt


# See a few example rows:


