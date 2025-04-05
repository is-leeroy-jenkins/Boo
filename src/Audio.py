'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                Audio.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2023

      Last Modified By:        Terry D. Eppler
      Last Modified On:        06-01-2023
  ******************************************************************************************
  <copyright file="Audio.py" company="Terry D. Eppler">

     This is a Federal Budget Execution and Data Analysis Application for EPA Analysts
     Copyright ©  2024  Terry Eppler

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
    Audio.py
  </summary>
  ******************************************************************************************
  '''
from typing import Any, Dict, List, Optional

import numpy as np
import PySimpleGUI as sg
import sounddevice as sd
import soundfile as sf
from loguru import logger

from src.Configuration import OUTPUT_FILE_NAME, SAMPLE_RATE

def find_blackhole_device_id() -> Optional[ int ]:
    '''
    Find the BlackHole device ID in the list of devices.

    Returns:
        Optional[int]: The BlackHole device ID if found, None otherwise.
    '''
    devices: List[ Dict[ str, Any ] ] = sd.query_devices()
    for device_id, device in enumerate( devices ):
        if 'BlackHole' in device[ 'name' ]:
            return device_id

    return None

def record( button: sg.Element ) -> None:
    '''
    Record audio from the BlackHole device while the record button is active.
    Save the audio to a file.

    Args:
        button (sg.Element): The record button element.
    '''
    logger.debug( 'Recording...' )
    frames: List[ np.ndarray ] = [ ]

    # Find BlackHole device ID
    device_id: Optional[ int ] = find_blackhole_device_id()

    # Record audio
    try:
        with sd.InputStream( samplerate = SAMPLE_RATE, device = device_id ) as stream:
            while button.metadata.state:
                data: np.ndarray
                overflowed: bool
                data, overflowed = stream.read( SAMPLE_RATE )
                if overflowed:
                    logger.warning( 'Audio buffer overflowed' )
                frames.append( data )

    except Exception as e:
        logger.error( f'An error occurred during recording: {e}' )

    # Save audio file
    if frames:
        audio_data: np.ndarray = np.vstack( frames )
        save_audio_file( audio_data )
    else:
        logger.warning( 'No audio recorded.' )

def save_audio_file(
        audio_data: np.ndarray, output_file_name: str = OUTPUT_FILE_NAME
) -> None:
    '''
    Save the audio values to a file.

    Args:
        audio_data (np.ndarray): The audio values.
        output_file_name (str, optional): The output file name. Defaults to OUTPUT_FILE_NAME.
    '''
    sf.write(
        file = output_file_name,
        data = audio_data,
        samplerate = SAMPLE_RATE,
        format = 'WAV',
        subtype = 'PCM_16',
    )
    logger.debug( f'Audio saved to: {output_file_name}...' )
