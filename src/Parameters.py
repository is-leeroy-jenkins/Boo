'''
Module defining classes used to encapsulate GPT parameter objects
'''
class GptParameter( ):
	'''
	The base class used by all parameter classes.
	'''
	def __init__( self, num: int = 1, temp: float = 0.11, top: float = 0.11, freq: float = 0.0,
	              pres: float = 0.0, store: bool = False, stream: bool = True ):
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.store = store
		self.stream = stream
		
class TextParameter( GptParameter ):
	'''
	Class to encapsulate the GPT Chat GptOptions object.
	'''
	def __init__( self, num: int = 1, temp: float = 0.11, top: float = 0.11, freq: float = 0.0,
	              pres: float = 0.0, store: bool = False, stream: bool = True ):
		super( ).__init__( num, temp, top, freq, pres, store, stream )
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.store = store
		self.stream = stream
		
class ChatParameter( GptParameter ):
	'''
	Class to encapsulate the GPT Chat GptOptions object.
	'''
	def __init__( self, num: int = 1, temp: float = 0.11, top: float = 0.11, freq: float = 0.0,
	              pres: float = 0.0, store: bool = False, stream: bool = True ):
		super( ).__init__( num, temp, top, freq, pres, store, stream )
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.store = store
		self.stream = stream
		
class EmbeddingParameter( GptParameter ):
	'''
	Class to encapsulate the GPT Chat GptOptions object.
	'''
	def __init__( self, num: int = 1, temp: float = 0.11, top: float = 0.11, freq: float = 0.0,
	              pres: float = 0.0, store: bool = False, stream: bool = True ):
		super( ).__init__( num, temp, top, freq, pres, store, stream )
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.store = store
		self.stream = stream
		
class FineTuningParameter( GptParameter ):
	'''
	Class to encapsulate the GPT Chat GptOptions object.
	'''
	def __init__( self, num: int = 1, temp: float = 0.11, top: float = 0.11, freq: float = 0.0,
	              pres: float = 0.0, store: bool = False, stream: bool = True ):
		super( ).__init__( num, temp, top, freq, pres, store, stream )
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.store = store
		self.stream = stream
		
class FileParameter( GptParameter ):
	'''
	Class to encapsulate the GPT Chat GptOptions object.
	'''
	def __init__( self, num: int = 1, temp: float = 0.11, top: float = 0.11, freq: float = 0.0,
	              pres: float = 0.0, store: bool = False, stream: bool = True ):
		super( ).__init__( num, temp, top, freq, pres, store, stream )
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.store = store
		self.stream = stream
		
class ImageParameter( GptParameter ):
	'''
	Class to encapsulate the GPT Chat GptOptions object.
	'''
	def __init__( self, num: int = 1, temp: float = 0.11, top: float = 0.11, freq: float = 0.0,
	              pres: float = 0.0, store: bool = False, stream: bool = True ):
		super( ).__init__( num, temp, top, freq, pres, store, stream )
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.store = store
		self.stream = stream
		
class SpeechParameter( GptParameter ):
	'''
	Class to encapsulate the GPT Chat GptOptions object.
	'''
	def __init__( self, num: int = 1, temp: float = 0.11, top: float = 0.11, freq: float = 0.0,
	              pres: float = 0.0, store: bool = False, stream: bool = True ):
		super( ).__init__( num, temp, top, freq, pres, store, stream )
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.store = store
		self.stream = stream
		
class TranslationParameter( GptParameter ):
	'''
	Class to encapsulate the GPT Chat GptOptions object.
	'''
	def __init__( self, num: int = 1, temp: float = 0.11, top: float = 0.11, freq: float = 0.0,
	              pres: float = 0.0, store: bool = False, stream: bool = True ):
		super( ).__init__( num, temp, top, freq, pres, store, stream )
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.store = store
		self.stream = stream
		
class TranscriptionParameter( GptParameter ):
	'''
	Class to encapsulate the GPT Chat GptOptions object.
	'''
	def __init__( self, num: int = 1, temp: float = 0.11, top: float = 0.11, freq: float = 0.0,
	              pres: float = 0.0, store: bool = False, stream: bool = True ):
		super( ).__init__( num, temp, top, freq, pres, store, stream )
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.store = store
		self.stream = stream
		
class VectorParameter( GptParameter ):
	'''
	Class to encapsulate the GPT Chat GptOptions object.
	'''
	def __init__( self, num: int = 1, temp: float = 0.11, top: float = 0.11, freq: float = 0.0,
	              pres: float = 0.0, store: bool = False, stream: bool = True ):
		super( ).__init__( num, temp, top, freq, pres, store, stream )
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.store = store
		self.stream = stream
		