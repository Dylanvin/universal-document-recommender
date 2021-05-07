# universal-document-recommender

*The following instructions are for starting the website on a PC*

This version is to be run on a PC. BERT is enabled on this version and the evaluation method 'evaluate_methods.py' is available.

Creating a Virtual Environment and Installing Libraries:
*It is recommended that a virtual environment is set up to install the libraries.*
1. To create the virtual environment run the following command.
          'python3 -m venv venv'

2. Then to activate the virtual environment run.
          'env\Scripts\activate'

3. To install the necessary libraries execute the following command.
          'pip install -r requirements.txt'

starting server:

1. To start the server run.
          'python app.py'


Using evaluation method:

1. Use the file datafiles/test_docs.txt to enter query URLs and the predicted category in the format URL,category.

2.The allowed categories are:
'alt.atheism',
'comp.graphics',
'comp.os.ms-windows.misc',
'comp.sys.ibm.pc.hardware',
'comp.sys.mac.hardware',
'comp.windows.x',
'misc.forsale',
'rec.autos',
'rec.motorcycles',
'rec.sport.baseball',
'rec.sport.hockey',
'sci.crypt',
'sci.electronics',
'sci.med',
'sci.space',
'soc.religion.christian',
'talk.politics.guns',
'talk.politics.mideast',
'talk.politics.misc',
'talk.religion.misc'

3. After adding the test documents run the command.
        'python evaluate_methods.py'

The resulting figures can be found in the figs folder.
