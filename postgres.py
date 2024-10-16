import psycopg2
import pandas as pd
import numpy as np
from pydantic_settings import BaseSettings
import logging
from psycopg2.extras import execute_values

LOGGER = logging.getLogger(__name__)

class AppSettings(BaseSettings):
    DBNAME: str
    USER: str
    PASSWORD: str
    HOST: str

class PostgresConnector:
    """a class that handles all backend function involved with the data in databases stored in Postgres"""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._connection = self._connect(self.settings)
        LOGGER.info(self._connection)

    def _connect(self, settings: AppSettings):
        try:
            dbname = settings.DBNAME
            user = settings.USER
            password = settings.PASSWORD
            host = settings.HOST
            return psycopg2.connect(
                user=user,
                password=password,
                host=host,
                database=dbname,
            )
        except Exception as e:
            LOGGER.error("Error connecting: %s", e)
            raise

    def get_cursor(self):
        """
        Get a cursor from the database connection.
        Reconnect if the connection is closed.
        """
        try:
            # Try to create a cursor
            cursor = self._connection.cursor()
        except (psycopg2.InterfaceError, psycopg2.OperationalError) as e:
            LOGGER.warning("Connection lost, attempting to reconnect: %s", e)
            # Reconnect if the connection has been closed or dropped
            self._connection = self._connect(self.settings)
            cursor = self._connection.cursor()
        return cursor

    def rollback(self):
        """Rollback the current transaction."""
        try:
            self._connection.rollback()
            LOGGER.info("Transaction rolled back")
        except Exception as e:
            LOGGER.error("Error during rollback: %s", e)
            raise

    def close(self):
        """Closes the database connection."""
        if self._connection:
            self._connection.close()
            LOGGER.info("Database connection closed")

    def query(self, query: str, parameters: tuple = None):
        cursor = self.get_cursor()

        # Use parameterized query if parameters are provided
        if parameters:
            cursor.execute(query, parameters)
        else:
            cursor.execute(query)

        self._connection.commit()
        return cursor

    def read(self, query: str, parameters: tuple = None):
        try:
            cursor = self.query(query, parameters)
            field_names = [i[0] for i in cursor.description]
            response = []
            for row in cursor:
                record = dict(zip(field_names, row))
                response.append(record)
        except Exception as e:
            self.rollback()  # Rollback the transaction if an error occurs
            raise psycopg2.OperationalError(f"Failed to execute query: {e}") from e
        finally:
            self.close()  # Ensure that the connection is closed in any case
        return response

    def write(self, query: str, values=None):
        connection = self._connection
        try:
            cursor = connection.cursor()
        except Exception:  # Reconnect if the connection has been closed
            connection = self._connect(self.settings)
            cursor = connection.cursor()
        try:
            # Execute the query
            if values is None:
                cursor.execute(query)
            else:
                execute_values(cursor, query, values)

            if "RETURNING" in query:
                records = cursor.fetchall()
            else:
                records = None

            connection.commit()
            self.close()
        except Exception as e:
            LOGGER.error("Error writing: %s", e)
            connection.rollback()
            raise

        return records