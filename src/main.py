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
