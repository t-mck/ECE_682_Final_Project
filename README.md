# ECE_682_Final_Project

## Overview project current state (10 min)  

We intend to recommend restaurants to individuals. Initially, our data source is the Kaggle Yelp dataset. 

- Text parsing – predict stars, maybe if things get serious, predict health score

  - Approaches:

    - Surveying 
    - NN -> needs a lot of prelabeled data (e.g. IMDB database?), but easier to implement (input review, label health code) (use SF kaggle dataset train) 

- Cluster businesses by features -> predict rating
- Cluster users -> recommend restaurants liked by other users in the cluster
- Regress user reviews of restaurants -> predict restaurant scores
- Regress over community (30 closest users) on restaurant score(avg of 30 users) to predict score
- Generate random forests to predict restaurant score for a particular user. 

## Idea generation regarding information gathering (20 min) 

How much of the data do we need to employ? I recommend just enough to get the job done – that is, start small and increment. Taylor said 50k user reviews and pair them with the health code, Julia said the SF health code dataset has ~50k 

A second implementation of the NLP stuff is to predict some feature between the user and a restaurant. However, we require idea generation on the features to predict (response). 

Which variables need to be in the user dataset? 
Which variables need to be in the business dataset? 

## Timeline + task breakdown (10 min)  

### Timeline
#### (1) must be done by Monday March 27th 
#### (2) must be done by Monday April 3rd 
#### (3) Must be done by Monday April 10th 
#### (4) must be done by Thursday April 13th 
#### (5) must be done by Wednesday April 19th 

### Tasks
- #### (1) Data wrangling  
  - Build business dataset   
  - Build user dataset  
- #### (1) Text parsing  
  - Build review -> health code dataset   
  - (2) Star prediction  
  - (2) Health Inspection Rating  
- (2) Cluster by business 
- (2) Cluster by users  
- (2) Regress per user  
- (2) Random forest  
- (3) Regress for community 
- (3.67) Analyze the model 
- (4) Ensemble somehow 
- (1-5) Report!!!!!! 
  - Technical appendix 
  - PR – 1page 
  - FAQ – 1-4pages 

## Hand out the work (10 min)  

### FOR (1)  
- Emma – Build business dataset 
- Taylor – Text Parsing 
- Julia – Build business dataset 
- Dev – Text Parsing 
- Andres – build user dataset 
- Mac – Build user dataset  
- Mac - will do all the random forest stuff. 