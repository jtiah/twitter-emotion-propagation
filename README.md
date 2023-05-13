# bsc-project: Emotion Propagation in Online Social Networks

This repo is the code and data used in our batchelor project, Emotion Propagation in Online Social Networks, and can be used to both expand and replicate. To comply with Twitter policy (see https://developer.twitter.com/en/developer-terms/agreement-and-policy) we don't provide the raw twitter texts, as we are only allowed to share ID's. If you need the full data, write the group at {gifa, jtih, joto}@itu.dk for full acces. 

The repo is structured into 3 main sections: code, data and other. 

### Code
Code contains the code used to run the project. The main results of the project can be checked by running the jupytor notebook "Workflow.ipynb", where most of the results of the projects can be found. Code also features multiple diffirent sub-folders that covers diffirent topics:
- workflow.ipynb: The main notebook with all code to run the project. As this repo is limited by the amount of data allowed published by Twitter, some of the code will not run, but are provided as source for what we did. 

- Libraries: Libraries of functions to for the workflow to use, to hide the complexity.
	- network_distance.py: Code by (Coscia, 2020) for finding the distance in a network. See https://www.michelecoscia.com/wp-content/uploads/2020/03/FULL-CosciaM1099.pdf for source.
	- utils.py: All the functions used in the workflow.



### Data
The data used. As we cannot provide the text of tweets, but only author_id and tweet_id, certain data is missing. To get accces to the full data, write the authors of the study. The data folder contains: "0- Raw collected data", "1- proccesed data", "2- NLP", "3- Propagation". 
The content of each folder is: 
- 0- Raw collected data: 
	- list-id.txt: File with the id number of all the lists
	- {network}: Folder with files:
		- DF-{network}-aux-retweets.csv: The original text of all retweets (OBS. Not in public repo)
		- DF-{network}-follows.csv: All the users followed, with some metadata on the followed user (OBS. Limited version in repo)
		- DF-{network}-tweets.csv: All tweets by user, with metadata associated with the tweets (OBS. Not in public repo)
		- DF-{network}-user_info.csv: Info about all users, including what sub-group they belong to (OBS. Not in public repo)


- 1- Proccesed data: Contains the subset of data used in the project, as well as network properties
	- DF-{network}-tweets.csv: The tweet text, cration time, author_id and if it was a retweet(OBS. not in public repr)
	- DF-{network}-user_info.csv: id, name, username, tweet_count, and subgroup for all users (OBS. Not in public repo)
	- edgelist-{network}-undirected.csv: Edges and their weights for network.
	- GRAPH: Gephi file with the 3 networks as graphs. (Might not work with limited data in public repo)


- 2- NLP: Folder for all NLP: Fine tuning and emotion extraction
	- 1- Fine Tuning: Folder for fine-tuning the model
		- Results.txt: Summary statistics for sampling of all networks
		- annotations: Folder with all of our annotation batches, the labels we gave, as well as the gold label.
			- Annotation guide: Our annotation guide
			- Batch {n} - Samples: The samples in batch n
			- Batch {n} - Summary: Our annotation of batch n, our agreed upon gold label and statistics for the batch.
			- Sampler.ipynb: Class to get sampels. 
		- {network/all}: Folder with test and train samples for the specific model. 
	- 2- Emotions: Folder with the emotions for all tweets
		- {network}-tweets-emotions.csv: The prediced value for all emotions with the author_id, created_at and nEntropy
		
- 3- Propagation: The distances each emotions have traveled for each day
	-{network}-distances.csv: The distances for each emoiton for each week for both normalized and non-normalized distance
























