'''
  ******************************************************************************************
      Assembly:                Boo
      Filename:                embbr.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2023

      Last Modified By:        Terry D. Eppler
      Last Modified On:        06-01-2023
  ******************************************************************************************
  <copyright file="tiggr.py" company="Terry D. Eppler">

     This is a Budget Execution and Data Analysis Application for Federal Analysts
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
  '''
import os
import re
import json
import pandas as pd
import re
import fitz
import string
import spacy
from openai import OpenAI
from booger import Error, ErrorDialog
from pathlib import Path
import tiktoken
import sqlite3
from typing import Any, List, Tuple, Optional


class Data( ):
    """
    DataAccess provides CRUD operations for a SQLite database.

    Methods:
        - create_table: Creates a table with specified schema.
        - insert: Inserts a record into a table.
        - fetch_all: Fetches all records from a table.
        - fetch_one: Fetches a single record matching the query.
        - update: Updates records that match a given condition.
        - delete: Deletes records that match a given condition.
        - close: Closes the database connection.
    """

    def __init__( self, db_name: str="Data.db" ):
        """
        Initializes the connection to the SQLite database.
        
        Args:
            db_name (str): The name of the database file.
        """
        self.conn = sqlite3.connect( db_name )
        self.cursor = self.conn.cursor( )

    def create_table( self, table_sql: str ) -> None:
        """
        Creates a table using a provided SQL statement.

        Args:
            table_sql (str): The CREATE TABLE SQL statement.
        """
        self.cursor.execute( table_sql )
        self.conn.commit( )

    def insert(self, table: str, columns: List[ str ], values: Tuple[Any, ...] ) -> None:
        """
        Inserts a new record into a table.

        Args:
            table (str): The name of the table.
            columns (List[str]): Column names.
            values (Tuple): Corresponding values.
        """
        placeholders = ", ".join( "?" for _ in values )
        col_names = ", ".join( columns )
        sql = f"INSERT INTO {table} ({col_names}) VALUES ({placeholders})"
        self.cursor.execute( sql, values )
        self.conn.commit()

    def fetch_all(self, table: str) -> List[Tuple]:
        """
        Retrieves all records from a table.

        Args:
            table (str): The name of the table.
        
        Returns:
            List[Tuple]: List of rows.
        """
        self.cursor.execute(f"SELECT * FROM {table}")
        return self.cursor.fetchall()

    def fetch_one(self, table: str, where_clause: str, params: Tuple[Any, ...]) -> Optional[Tuple]:
        """
        Retrieves a single row matching a WHERE clause.

        Args:
            table (str): Table name.
            where_clause (str): WHERE clause (excluding 'WHERE').
            params (Tuple): Parameters for the clause.
        
        Returns:
            Optional[Tuple]: The fetched row or None.
        """
        sql = f"SELECT * FROM {table} WHERE {where_clause} LIMIT 1"
        self.cursor.execute(sql, params)
        return self.cursor.fetchone()

    def update(self, table: str, set_clause: str, where_clause: str, params: Tuple[Any, ...]) -> None:
        """
        Updates records in a table.

        Args:
            table (str): Table name.
            set_clause (str): SET clause with placeholders.
            where_clause (str): WHERE clause with placeholders.
            params (Tuple): Parameters for both clauses.
        """
        sql = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
        
        self.cursor.execute(sql, params)
        self.conn.commit()

    def delete(self, table: str, where_clause: str, params: Tuple[Any, ...]) -> None:
        """
        Deletes records matching the given WHERE clause.

        Args:
            table (str): Table name.
            where_clause (str): WHERE clause (excluding 'WHERE').
            params (Tuple): Parameters for clause.
        """
        sql = f"DELETE FROM {table} WHERE {where_clause}"
        self.cursor.execute(sql, params)
        self.conn.commit()

    def close(self) -> None:
        """
        Closes the database connection.
        """
        self.conn.close()

