from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing import image
import socket
import requests
import json
from socket import *
import pickle
import psycopg2
import datetime
import requests

# Step 1 - Create a TCP/IP socket.
# This socket is used to connect to any TCP/IP socket server
# Code for server in server.py
clnt_sock = socket(AF_INET,SOCK_STREAM)

# #Step 2 - Connect the socket to the server
clnt_sock.connect(('192.168.43.231',3980))
print "connected"
#Step 3 - Receive the welcome message from the serverr
msg = clnt_sock.recv(1024)
print(msg)

#Code to get GeoLocation of Traffic Camera
addressRequest = 'http://ip-api.com/json/122.164.114.11'
address = requests.get(addressRequest)
location = json.loads(address.text)
lat = location['lat']
lon = location['lon']
regionName = location['regionName']
pinCode = location['zip']
ts = datetime.datetime.now()


#Database Connection & Table Creation
def connectToDb():
	#connecting to DB
	global conn
	conn = psycopg2.connect(database="majorProject", user = "postgres", 
		password = "pa$$word", host = "localhost", port = "5432")
	print "Opened database successfully"

	# #creating table
	# cur = conn.cursor()
	# cur.execute('''CREATE TABLE AccidentList (
	# 	accidentId SERIAL PRIMARY KEY,
	# 	latitude VARCHAR,
	# 	longitude VARCHAR,
	# 	accidentTimeStamp TIMESTAMPTZ);''')
	# cur.execute('''SET timezone = 'Asia/Kolkata';''')
	# print "Table created successfully"

connectToDb()

#creating other tables
def createAllTables():

	cur = conn.cursor()

	cur.execute('''CREATE TABLE Rsu(
		rsuId SERIAL PRIMARY KEY,
		areaCode VARCHAR,
		UNIQUE(areaCode));''')

	cur.execute('''CREATE TABLE Vanet (
	vanetId SERIAL PRIMARY KEY REFERENCES Rsu(rsuId),
	areaCode VARCHAR);''')

	cur.execute('''CREATE TABLE VanetVehicle (
	vehicleId SERIAL PRIMARY KEY,
	vanetId INT REFERENCES Vanet(vanetId));''')

	cur.execute("ALTER TABLE AccidentList ADD COLUMN areaCode INT");

# createAllTables()

#Query to intert values to table if accident occurs
def insertIntoAccidentTable():
	cur = conn.cursor()

	#inserting values
	query = "INSERT INTO AccidentList (latitude,longitude,accidenttimestamp) VALUES(%s, %s, %s);"
	data = (lat,lon,ts)
	cur.execute(query,data)
	print "Values Inserted"

	conn.commit()
	conn.close() 

def sendSMS():
	url = "https://www.fast2sms.com/dev/bulk"
	payload = "sender_id=FSTSMS&message=ACCIDENT HAS OCCURED AT LAT: 13.0843, LON: 80.2805 &language=english&route=p&numbers=8667485230,7299919794"
	headers = {
		'authorization': "KafuRvXhnZWL4bkTjrqSlHcVs39UAyd8xMi157CQFNPEBwzIeDRwm32Z84OcPosSNn9WDKpiBIULyv50",
		'Content-Type': "application/x-www-form-urlencoded",
		'Cache-Control': "no-cache",
	}
	response = requests.request("POST", url, data=payload, headers=headers)
	print(response.text)

#Machine Learning Algorithm  
classifier=load_model('trainTRY.h5')
test_image = cv2.imread('309.jpg')
test_image = cv2.resize(test_image,(64,64))
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
result1=np.argmax(result)
print(result)
print(result1)

prediction=""
if result1==  1:
	prediction = "Accident Occurred"
	insertIntoAccidentTable()
	#sendSMS()
elif result1 == 2:
	prediction = "Fire"
else:
  prediction='No Accident Occurred'
print prediction

print ("Latitude: "+str(lat)+" \nLongitutde: "+str(lon)+"\nRegion: "+regionName+
	"\nPincode: "+pinCode+"\nTimeStamp: "+str(ts))

clnt_sock.send(prediction)
clnt_sock.send(str(lat))
clnt_sock.send(str(lon))
clnt_sock.send(pinCode)

clnt_sock.close()