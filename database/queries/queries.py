import logging 

LOGGER = logging.getLogger(__name__)

def insert_query(client, table_name, cols, values):
    try:
        sql_statement = "INSERT INTO %s(%s) VALUES %%s" % (table_name, cols)
        result = client.write(
            query=sql_statement,
            values=values,
        )
    except Exception as e:
        # Log the error, close the client, and raise an HTTP 400 error if writing to the database fails
        LOGGER.info(e)
        client.close()
        raise
    return result

def delete_query(client, table_name, condition, value):
    try: 
        sql_statement = "DELETE FROM %s WHERE %s = %s" % (table_name, condition, value)
        result = client.write(
            query=sql_statement
        )
    except Exception as e:
        # Log the error, close the client, and raise an HTTP 400 error if writing to the database fails
        LOGGER.info(e)
        client.close()
        raise
    return result

def read_query(client, sql_statement, params): 
    try: 
        result = client.read(
            query=sql_statement, 
            parameters = params
        )
    except Exception as e:
        # Log the error, close the client, and raise an HTTP 400 error if writing to the database fails
        LOGGER.info(e)
        client.close()
        raise
    return result