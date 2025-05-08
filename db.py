import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  database="plaka_tanima",
  password=""
)

print(mydb)
