from flask_wtf import FlaskForm
from wtforms import StringField, RadioField, TextAreaField, SelectField
from wtforms.validators import DataRequired, Length

class RunSystemForm(FlaskForm):
    #query = StringFiled('query', validators=[DataRequired(), Length(min=1, max=200)])
    algorithm =  RadioField(
        'Algorithm selection', choices=[('TfIdf', 'Tf-Idf'), ('lsa', 'LSA'), ('Doc2Vec', 'Doc2Vec')])
    query = TextAreaField('Document text', validators=[DataRequired()])
    category = SelectField("Choose an option")