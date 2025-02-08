'''
Module containing the GPT reqeusts for the application
'''
class GptRequest( ):
	'''
	Base class for GPT requests.
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

class TextGenerationRequest( GptRequest ):
	'''
	Class encapsulating the request for text generations.
	'''
	def __init__( self, num: int = 1, temp: float = 0.18, top: float = 0.11, freq: float = 0.0,
	              pres: float = 0.0, store: bool = False, stream: bool = True ):
		super( ).__init__( num, temp, top, freq, pres, store, stream )
		self.number = num
		self.temperature = temp
		self.top_percent = top
		self.frequency_penalty = freq
		self.presence_penalty = pres
		self.store = store
		self.stream = stream
	
class ChatCompletionRequest( GptRequest ):
	'''
	Class encapsulating requests for chat completions.
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

class SpeechGenerationRequest( GptRequest ):
	'''
	Class encapsulating requests for speech generations.
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

class TranslationRequest( GptRequest ):
	'''
	Class encapsulating requests for translation.
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
	
class TranscriptionRequest( GptRequest ):
	'''
	Class encapsulating requests for transcriptions.
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

class EmbeddingRequest( GptRequest ):
	'''
	Class encapsulating requests for embedding.
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

class VectorRequest( GptRequest ):
	'''
	Class encapsulating requests for vectors.
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

class GptFileRequest( GptRequest ):
	'''
	Class encapsulating requests for GPT files.
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

class GptUploadRequest( GptRequest ):
	'''
	Class encapsulating requests for GPT uploads.
	'''

class FineTuningRequest( GptRequest ):
	'''
	Class encapsulating requests for fine-tuning.
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

class ImageGenerationRequest( GptRequest ):
	'''
	Class encapsulating requests for image generation.
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
		