# universal-document-recommender

*The following instructions are for starting the website on a Raspberry Pi*

This version is designed to work on the Raspberry Pi 3 Model B V1.2 by disabling BERT and may or may not work on other Raspberry Pi models. The evaluation method
is not in this version.

Creating a Virtual Environment and Installing Libraries:
*It is recommended that a virtual environment is set up to install the libraries.*
1. To create the virtual environment run the following command.
          'sudo python3 -m venv venv'

2. Then to activate the virtual environment run.
          'source env/bin/activate'

3. To install the necessary libraries execute the following command.
          'pip install -r requirements.txt'

Setting Up Flask Environment Variable:

1. run the following command.
          'export FLASK_APP=app.py'

2. This means the server can now be started by using the command:
          'flask run'

3. To host the server or port 5000 from and allow it to be accessed from any IP
on your machine use the command.
          'flask run -h 0.0.0.0'

You will need to port forward in order to allow machines on external networks to
see the website.
