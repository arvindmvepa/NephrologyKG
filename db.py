import mysql.connector


def connect_db(host='localhost', user='root', password='', db='umls'):
    return mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        db=db)
