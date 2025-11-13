'''
******************************************************************************************
  Assembly:                Boo
  Filename:                forms.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2022

  Last Modified By:        Terry D. Eppler
  Last Modified On:        05-01-2025
******************************************************************************************
	<copyright file="forms.py" company="Terry D. Eppler">
		     forms.py
		     Copyright ©  2022  Terry Eppler
		
		 Permission is hereby granted, free of charge, to any person obtaining a copy
		 of this software and associated documentation files (the “Software”),
		 to deal in the Software without restriction,
		 including without limitation the rights to use,
		 copy, modify, merge, publish, distribute, sublicense,
		 and/or sell copies of the Software,
		 and to permit persons to whom the Software is furnished to do so,
		 subject to the following conditions:
		
		 The above copyright notice and this permission notice shall be included in all
		 copies or substantial portions of the Software.
		
		 THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
		 INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
		 FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
		 IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
		 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
		 ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
		 DEALINGS IN THE SOFTWARE.
		
		 You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov
	</copyright>
<summary>
	forms.py
</summary>
******************************************************************************************
'''
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from flask import Flask, render_template, redirect, url_for, flash, request
from flask_wtf import FlaskForm, CSRFProtect
from wtforms import (
    StringField, PasswordField, BooleanField, SubmitField, TextAreaField,
    SelectField, FormField
)
from wtforms.validators import DataRequired, Email, Length, EqualTo, Optional, Regexp
from flask_wtf.file import FileField, FileAllowed, FileRequired
from werkzeug.utils import secure_filename
from wtforms import Form  # base class for nested sub-forms
import os
from pathlib import Path

class NameForm( FlaskForm ):
	name = StringField( 'What is your name?', validators=[ DataRequired( ) ] )
	submit = SubmitField( 'Submit' )
	
class LoginForm(FlaskForm):
    username = StringField( 'Username', validators=[ DataRequired( ), Length( min=3, max=64 ) ] )
    password = PasswordField( 'Password', validators=[ DataRequired( ) ] )
    remember = BooleanField( 'Remember me' )
    submit = SubmitField( 'Login' )

class RegisterForm( FlaskForm ):
    email = StringField( 'Email', validators=[ DataRequired( ), Email()])
    username = StringField( 'Username', validators=[ DataRequired( ), Length( min=4, max=20 ) ] )
    password = PasswordField( 'Password', validators=[ DataRequired( ), Length( min=6 ) ] )
    confirm  = PasswordField( 'Confirm', validators=[ DataRequired( ), EqualTo( 'password' ) ] )
    role = SelectField( 'Role', choices=[ ( 'user','User' ), ( 'admin','Admin' ) ], validators=[ DataRequired( ) ] )
    submit = SubmitField( 'Create Account' )

class UploadForm( FlaskForm ):
    title = StringField( 'Title', validators=[ DataRequired( ), Length( max=80 ) ] )
    doc = FileField(
        'Image/PDF',
        validators=[ FileRequired( ), FileAllowed( [ 'png','jpg','jpeg','gif','pdf' ], 'Images/PDFs only!' ) ], )
    submit = SubmitField( 'Upload' )

# Nested sub-form uses plain `Form`
class AddressSubForm( Form ):
    line1 = StringField( 'Address Line 1', validators=[ DataRequired( ), Length( max=100 ) ] )
    city  = StringField( 'City', validators=[ DataRequired( ), Length( max=50 ) ] )
    state = StringField( 'State', validators=[ DataRequired( ), Length( min=2, max=2 ) ] )
    zip   = StringField( 'ZIP',  validators=[ DataRequired( ), Regexp( r'^\d{5}(-\d{4})?$')] )

class ProfileForm( FlaskForm ):
    full_name = StringField( 'Full Name', validators=[ DataRequired( ), Length( max=80 ) ] )
    bio = TextAreaField( 'Bio', validators=[ Optional( ), Length( max=300 ) ] )
    address = FormField( AddressSubForm )
    submit = SubmitField( 'Save Profile' )

# Tiny feedback form for the multi-form example:
class FeedbackOnSamePage(FlaskForm):
    category = SelectField( 'Category', choices=[ ( 'bug','Bug' ),( 'feature','Feature' ),( 'other','Other' ) ] )
    message = TextAreaField( 'Message', validators=[ DataRequired( ), Length( min=10) ] )
    submit = SubmitField( 'Send' )