'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                main.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="main.py" company="Terry D. Eppler">

         Boo is a df analysis tool integrating GenAI, GptText Processing, and Machine-Learning
         algorithms for federal analysts.
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
    main.py
    
  </summary>
  ******************************************************************************************
'''
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from flask import (Flask, render_template, session, request, redirect, current_app, abort, url_for, flash)
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from datetime import datetime
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import config
from controls import NameForm

app = Flask( __name__ )
app.config[ 'SECRET_KEY' ] = config.SECRET_KEY
app.config[ 'SQLALCHEMY_DATABASE_URI' ] = 'sqlite:///' + config.SQLALCHEMY_DATABASE_URI
app.config[ 'SQLALCHEMY_TRACK_MODIFICATIONS' ] = False
bootstrap = Bootstrap( app )
moment = Moment( app )
db = SQLAlchemy(app)

@app.route( '/', methods=[ 'GET', 'POST' ] )
def index( ):
	form = NameForm( )
	if form.validate_on_submit( ):
		session[ 'name' ] = form.name.data
		return redirect( url_for( 'index' ) )
	return render_template( 'index.html', form=form, name=session.get( 'name' ) )
	return '<User %r>' % self.username
@app.route( '/user/<name>' )
def user( name ):
    return render_template( 'user.html', name=name )

@app.errorhandler( 404 )
def page_not_found( e ):
    return render_template( '404.html' ), 404

@app.errorhandler( 500 )
def internal_server_error( e ):
    return render_template( '500.html' ), 500

class Role( db.Model ):
	__tablename__ = 'roles'
	id = db.Column( db.Integer, primary_key=True )
	name = db.Column( db.String( 64 ), unique=True )
	users = db.relationship( 'User', backref='role' )
	
	def __repr__( self ):
		return '<Role %r>' % self.name

class User( db.Model ):
	__tablename__ = 'users'
	id = db.Column( db.Integer, primary_key=True )
	username = db.Column( db.String( 64 ), unique=True, index=True )
	role_id = db.Column( db.Integer, db.ForeignKey( 'roles.id' ) )
	
	def __repr__( self ):

if __name__ == '__main__':
	app.run( )
