'''
	Module defining the messages used in the GPT requests
'''

class GptMessage( ):
	'''
		Base class for all messages used in the GPT application
	'''
	def __init__( self, prompt: str ) -> None:
		self.content = prompt
		
class SystemMessage( GptMessage ):
	'''
	
		Class used to represent a system message.
	
	'''
	def __init__( self, prompt: str ):
		super( ).__init__( prompt )
		self.role = "system"
		self.prompt = prompt
		
	def __str__( self ):
		'''
		
			Returns: the json string representation of the message.

		'''
		if self.content is not None:
			_data = f" 'role': 'user', \r\n 'content': '{self.content}' "
			_message = "{ " + _data + " }"
			return _message
		
class UserMessage( GptMessage ):
	'''
	
		Class used to represent a user message.
	
	'''
	def __init__( self, prompt: str ):
		super( ).__init__( prompt )
		self.role = "user"
		self.content = prompt
		
	def __str__( self ):
		'''
		
			Returns: the json string representation of the message.

		'''
		if self.content is not None:
			_data = f" 'role': 'user', \r\n 'content': '{self.content}' "
			_message = "{ " + _data + " }"
			return _message
		
class AssistantMessage( GptMessage ):
	'''
	
		Class used to represent a assistant message.
	
	'''
	def __init__( self, prompt: str ):
		super( ).__init__( prompt )
		self.role = "assistant"
		self.content = prompt
		
	def __str__( self ):
		'''
		
			Returns: the json string representation of the message.

		'''
		if self.content is not None:
			_data = f" 'role': 'user', \r\n 'content': '{self.content}' "
			_message = "{ " + _data + " }"
			return _message


class ChatLog(  ):
	'''
	
	Class used to encapsulate a collection of chat messages.
	
	'''
	def __init__( self ):
		self.messages = []
		