from flask_wtf import FlaskForm
from wtforms import RadioField, TextAreaField, SelectField, IntegerField, StringField, BooleanField
from wtforms.validators import DataRequired, ValidationError
from wtforms.widgets import html5 as h5widgets
from urllib.parse import urlparse
import requests


class RunSystemForm(FlaskForm):
    algorithm = RadioField(
        'Algorithm selection', choices=[('TfIdf', 'Tf-Idf'), ('LSA', 'LSA'), ('Doc2Vec', 'Doc2Vec'), ('BERT', 'BERT')],
        default='TfIdf')
    # algorithm = RadioField(
    #     'Algorithm selection', choices=[('TfIdf', 'Tf-Idf'), ('LSA', 'LSA'), ('Doc2Vec', 'Doc2Vec')],
    #     default='TfIdf')
    measurement = RadioField(
        'Measurement selection', choices=[('cosine', 'Cosine'), ('euclidean', 'Euclidean')], default='cosine')
    query = TextAreaField('Document text', default='Enter query here')
    query_url = StringField('Document url')
    category = SelectField("Choose an option", validators=[DataRequired()])
    num = IntegerField(
        "num", widget=h5widgets.NumberInput(min=1, max=10), default=5, validators=[DataRequired()])
    showTextBox = BooleanField()
    eval_select = BooleanField()

    def validate_query_url(self, query_url):
        """
        validation method to ensure given URL is valid

        :param query_url:
        :return:
        """
        if not self.showTextBox.data:
            URL = query_url.data
            try:
                requests.get(URL).text
            except:
                raise ValidationError("invalid URL")
