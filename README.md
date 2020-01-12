# An Experimental Setup for Progressive Model Development Trained on the Toxic Comment Classification Dataset 

prepared by JPula (Jared Rivera)

# Libraries Needed

To run the file, the following packages are required.

* pandas
* re
* sklearn
* importlib

# Usage

* Only the "setup.py" python file must be run
* Should any problems be encounter with the raw dataset files, the .csv files may be found on the Data section of
  https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge. Simply download these files and add them to the data/00_raw/
  directory.
  

# Author's Note

While the author of this repository is aware that this is the simplest possible model setup to accomplish multi-label text classification, the project focused on creating a simple setup to motivate further development of the model such as:  
- an updated preprocessing script that considers the importance of punctuation and special characters
- TF-IDF Feature Selection to preserve meaningful information taken from the order of tokens
- a comparison between Multinomial Naive Bayes and SVMs
- etc.,

The author can only hope that the simplicity of the model will not affect the perception of his efforts to improve.
