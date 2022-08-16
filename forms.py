import wtforms
from wtforms.validators import length,email
class LoginForm(wtforms.Form):
	email=wtforms.StringField(validators=[length(min=5,max=20),email()])
	password=wtforms.StringField(validators=[length(min=6,max=20)])