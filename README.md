
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
    for f in *.txt; do (cat "${f}"; echo) >> positive_movie_reviews.txt; done

And

    cd ~/aclImdb/train/neg
    for f in *.txt; do (cat "${f}"; echo) >> negative_movie_reviews.txt; done


# See an example of a positive movie review:

    $ head -n 1 positive_movie_reviews.txt

"Bromwell High is a cartoon comedy. It ran at the same time as some other programs about school life, such as "Teachers". My 35 years in the teaching profession lead me to believe that Bromwell High's satire is much closer to reality than is "Teachers". The scramble to survive financially, the insightful students who can see right through their pathetic teachers' pomp, the pettiness of the whole situation, all remind me of the schools I knew and their students. When I saw the episode in which a student repeatedly tried to burn down the school, I immediately recalled ......... at .......... High. A classic line: INSPECTOR: I'm here to sack one of your teachers. STUDENT: Welcome to Bromwell High. I expect that many adults of my age think that Bromwell High is far fetched. What a pity that it isn't!"



# See an example of a negative movie review:

    $ head -n 1 negative_movie_reviews.txt

"Story of a man who has unnatural feelings for a pig. Starts out with a opening scene that is a terrific example of absurd comedy. A formal orchestra audience is turned into an insane, violent mob by the crazy chantings of it's singers. Unfortunately it stays absurd the WHOLE time with no general narrative eventually making it just too off putting. Even those from the era should be turned off. The cryptic dialogue would make Shakespeare seem easy to a third grader. On a technical level it's better than you might think with some good cinematography by future great Vilmos Zsigmond. Future stars Sally Kirkland and Frederic Forrest can be seen briefly."


# Install requirements and libraries: 

### Tensorflow

### python3

### keras

    sudo pip3 install keras


