from typing import Any, Dict
import FreeSimpleGUI as sg
from loguru import logger



def main( ) -> None:
    """

        Main function. Initialize the window and handle the events.

    """
    window = sg.Window( 'Hello World!' )
    logger.debug( 'Application started.' )

    while True:
        event: str
        values: Dict[ str, Any ]
        event, values = window.read( )

        if event in [ '-CLOSE_BUTTON-', sg.WIN_CLOSED ]:

            logger.debug( 'Closing...' )
            break
    window.close( )


if __name__ == "__main__":
    main( )
