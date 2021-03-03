from flask_wtf import FlaskForm
from wtforms import RadioField, TextAreaField, SelectField, IntegerField
from wtforms.validators import DataRequired
from wtforms.widgets import html5 as h5widgets

class RunSystemForm(FlaskForm):
    #query = StringFiled('query', validators=[DataRequired(), Length(min=1, max=200)])
    algorithm =  RadioField(
        'Algorithm selection', choices=[('TfIdf', 'Tf-Idf'), ('lsa', 'LSA'), ('Doc2Vec', 'Doc2Vec'), ('bert', 'BERT')], default='TfIdf')
    measurement = RadioField(
        'Measurement selection', choices=[('cosine', 'Cosine'), ('euclidean', 'Euclidean')], default='cosine')
    query = TextAreaField('Document text', validators=[DataRequired()], default='Enter query here')
    category = SelectField("Choose an option", validators=[DataRequired()])
    num = IntegerField(
        "num", widget=h5widgets.NumberInput(min=1, max=10), default=5, validators=[DataRequired()])