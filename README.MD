# One Step Ahead: A Framework for Detecting Unexpected Incidents and Predicting the Stock Markets

This is the core code for the paper : *One Step Ahead: A Framework for Detecting Unexpected Incidents and Predicting the Stock Markets*

## Environment Requirement

>pytorch == 1.2.0  
>scikit-learn == 0.22  
>numpy == 1.16.2

## Market prediction  
The [market_prediction.py](market_prediction.py) file contains the core code of the market prediction part:   
feature composition, model training and testing (historical model and one-step-ahead prediction). 

## Binary classification model

The [binary_classifier.py](binary_classifier.py) file contains the code of the deep learning model.

## Information Extraction

The [extractor.py](extractor.py) file contains the core of the information extraction.  
You can use the function *extract_casualties* to extract the casualties from the news text. The function *getlocation* in class *Loaction_extactor* is used to extract the information about the loaction which appears in the news title.
## Extra Data
You can obtain data from the following hyperlink:

>[Nightlight data](https://eoimages.gsfc.nasa.gov/images/imagerecords/144000/144897/BlackMarble_2016_3km_gray_geo.tif)  
[Underlying economic data](https://sedac.ciesin.columbia.edu/data/set/spatialecon-gecon-v4)  
[GTD data](https://www.start.umd.edu/gtd/)  
[The New York Times Developer Network](https://developer.nytimes.com/)  
[GloVe](https://nlp.stanford.edu/projects/glove/)  
[GeoNames geographical database](http://download.geonames.org/export/dump/)  

Capital / Cultural center / Religious center labels data From Wikipedia.  
Unfortunately, the Reuters archive website is not available now.  