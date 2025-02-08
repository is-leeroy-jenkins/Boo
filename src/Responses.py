'''
Module containing the GPT reqeusts for the application
'''
import datetime as dt

class GptResponse( ):
	'''
	Base class for GPT requests.
	'''
	def __init__( self, respid: str, obj: object, model: str, created: dt.datetime ):
		self.id = respid
		self.object = obj
		self.model = model
		self.created = created
		
class ChatCompletionResponse( GptResponse ):
	'''
	Class containing the GPT response for the chat completion
	'''
	def __init__( self, respid: str, obj: object, model: str, created: dt.datetime ):
		super( ).__init__( respid, obj, model, created )
		self.id = respid
		self.object = obj
		self.model = model
		self.created = created

class TextGenerationResponse( GptResponse ):
	'''
	Class containing the GPT response for the text generation
	'''
	def __init__( self, respid: str, obj: object, model: str, created: dt.datetime ):
		super( ).__init__( respid, obj, model, created )
		self.id = respid
		self.object = obj
		self.model = model
		self.created = created
	
class EmbeddingResponse( GptResponse ):
	'''
	Class containing the GPT response for the embedding request
	'''
	def __init__( self, respid: str, obj: object, model: str, created: dt.datetime ):
		super( ).__init__( respid, obj, model, created )
		self.id = respid
		self.object = obj
		self.model = model
		self.created = created

class FineTuningResponse( GptResponse ):
	'''
	Class containing the GPT response for the fine tuning request
	'''
	def __init__( self, respid: str, obj: object, model: str, created: dt.datetime ):
		super( ).__init__( respid, obj, model, created )
		self.id = respid
		self.object = obj
		self.model = model
		self.created = created

class VectorResponse( GptResponse ):
	'''
	Class containing the GPT response for the vector request
	'''
	def __init__( self, respid: str, obj: object, model: str, created: dt.datetime ):
		super( ).__init__( respid, obj, model, created )
		self.id = respid
		self.object = obj
		self.model = model
		self.created = created

class GptFileResponse( GptResponse ):
	'''
	Class containing the GPT response for the file request
	'''
	def __init__( self, respid: str, obj: object, model: str, created: dt.datetime ):
		super( ).__init__( respid, obj, model, created )
		self.id = respid
		self.object = obj
		self.model = model
		self.created = created
	
class GptUploadResponse( GptResponse ):
	'''
	References
	'''
	def __init__( self, respid: str, obj: object, model: str, created: dt.datetime ):
		super( ).__init__( respid, obj, model, created )
		self.id = respid
		self.object = obj
		self.model = model
		self.created = created

class ImageGenerationResponse( GptResponse ):
	'''
	Class containing the GPT response for the image request
	'''
	def __init__( self, respid: str, obj: object, model: str, created: dt.datetime ):
		super( ).__init__( respid, obj, model, created )
		self.id = respid
		self.object = obj
		self.model = model
		self.created = created
		