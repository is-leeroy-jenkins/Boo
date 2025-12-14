'''
  ******************************************************************************************
      Assembly:                Name
      Filename:                name.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="guro.py" company="Terry D. Eppler">

	     name.py
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
    name.py
  </summary>
  ******************************************************************************************
'''
import os
from datetime import datetime
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from flask import Flask, render_template, session, redirect, url_for, flash, request
from flask_sqlalchemy import SQLAlchemy

from forms import NameForm, LoginForm, RegisterForm, UploadForm, ProfileForm, FeedbackOnSamePage, ContactForm
import config
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
import config as cfg

app = Flask( __name__ )
app.config[ 'SECRET_KEY' ] = config.SECRET_KEY
app.config[ 'SQLALCHEMY_DATABASE_URI' ] = config.SQLALCHEMY_DATABASE_URI
bootstrap = Bootstrap( app )
moment = Moment( app )
db = SQLAlchemy( app )
CSRFProtect( app )

@app.route( '/', methods=[ 'GET', 'POST' ] )
def index( ):
    # Initialize chat history in session
    if 'chat_history' not in session:
        session[ 'chat_history' ] = [ ]

    if request.method == 'POST':
        user_msg = request.form.get( 'message', "" ).strip( )

        if user_msg:
            # Add user message
            session[ 'chat_history' ].append(
	        {
                'sender': 'user',
                'text': user_msg,
                'time': datetime.now().strftime( '%H:%M' )
            })

            # Dummy system response (Replace with OpenAI or custom logic)
            reply = f'You said: {user_msg}'

            session[ 'chat_history' ].append(
	        {
                'sender': 'assistant',
                'text': reply,
                'time': datetime.now( ).strftime( '%H:%M' )
            })

        # Save session
        session.modified = True

        return redirect(url_for( 'index' ) )

    return render_template( 'index.html', chat=session[ 'chat_history' ] )

@app.route( '/user/<name>' )
def user( name ):
    return render_template( 'user.html', name=name )

@app.route( '/login', methods=[ 'GET','POST' ] )
def login( ):
    form = LoginForm( )
    if form.validate_on_submit( ):
        flash( f'Logged in as {form.username.data}', 'success' )
        return redirect( url_for( 'index' ) )
    return render_template( 'login.html', form=form )

@app.route('/register', methods=[ 'GET','POST' ])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        flash(f'Registered user "{form.username.data}"', 'info' )
        return redirect( url_for( 'index' ) )
    return render_template( 'register.html', form=form )

@app.route( '/upload', methods=[ 'GET','POST' ] )
def upload( ):
    form = UploadForm( )
    if form.validate_on_submit( ):
        f = form.doc.data
        name = secure_filename( f.filename or '')
        f.save( os.path.join( app.config[ 'UPLOAD_FOLDER' ], name ) )
        flash( f'Uploaded "{form.title.data}" as {name}', 'success' )
        return redirect( url_for( 'index' ) )
    return render_template( 'upload.html', form=form )

@app.route( '/profile', methods=[ 'GET','POST'] )
def profile( ):
    form = ProfileForm( )
    if request.method == 'GET':
        # prefill example
        form.full_name.data = 'Brocifus Analyst'
        form.address.line1.data = '123 Market St'
        form.address.city.data = 'Arlington'
        form.address.state.data = 'VA'
        form.address.zip.data = '22202'
    if form.validate_on_submit( ):
        flash( 'Profile saved.', 'success' )
        return redirect( url_for( 'index' ) )
    return render_template( 'profile.html', form=form )

@app.route( '/multi', methods=[ 'GET','POST' ] )
def multi( ):
    login_form = LoginForm( prefix='login' )
    fb_form = FeedbackOnSamePage( prefix='fb' )  # defined inline below for brevity
    if request.method == 'POST':
        if 'login-submit' in request.form and login_form.validate_on_submit( ):
            flash( f'Multi-page login: {login_form.username.data}', 'success' )
            return redirect(url_for( 'multi' ) )
        if 'fb-submit' in request.form and fb_form.validate_on_submit( ):
            flash( f'Multi-page feedback: { fb_form.category.data }', 'success' )
            return redirect( url_for( 'multi' ) )
    return render_template( 'base.html' )  #

@app.route("/contact", methods=[ 'GET', 'POST' ] )
def contact():
    form = ContactForm()

    if form.validate_on_submit():
        name = form.name.data
        email = form.email.data
        subject = form.subject.data
        message = form.message.data

        # Here you could:
        # - Send an email
        # - Save to database
        # - Write to file/log
        # For now we just flash confirmation.

        flash( 'Your message has been sent successfully.' )
        return redirect(url_for( 'contact' ))

    return render_template( 'contact.html', form=form )

@app.errorhandler( 404 )
def page_not_found( e ):
	return render_template( '404.html' ), 404

@app.errorhandler( 500 )
def internal_server_error( e ):
	return render_template( '500.html' ), 500

class Role( db.Model ):
	__tablename__ = 'Roles'
	id = db.Column( db.Integer, primary_key=True )
	name = db.Column( db.String( 64 ), unique=True )
	users = db.relationship( 'User', backref='Role' )

class User( db.Model ):
	__tablename__ = 'Users'
	id = db.Column( db.Integer, primary_key=True )
	username = db.Column( db.String( 64 ), unique=True, index=True )
	role_id = db.Column( db.Integer, db.ForeignKey( 'Roles.id' ) )

	
if __name__ == '__main__':
	app.run( )
