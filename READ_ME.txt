
** The website is deployed online and can be accessed through the following link:


 
** Requirements for Easy Diagnosis flask application:

— Python version installed: 3.X , X>5

— Install MongoDB Compass

—  Then apply the following steps: 

1-  If Python version installed: 3.X , X>5 is not installed , install it.

2-  Install Anaconda and start a virtual environment using the following command in terminal or command prompt: 

		- conda create -n name_of_env python=3.X 

	then activate the environment by typing 

		 - conda activate name_of_env

##--important note: use pip3 instead of pip always if the python version was 3.X--##
-----------------------------------------------------------------------------------------

3- install all the app requirements by running the following command FIRST TIME :\\NEED THIS COMMAND JUST ONE TIME  
		
	pip install -r requirements.txt
-- OR USE FOR MAC --
	pip3 install -r requirements.txt

6-  Set the environment variable FLASK_APP to ED_APP.py
	
	for Mac OS:
		
		export NEWVAR=SOMETHING

EX: export FLASK_APP=ED_APP.py  //no spaces


	for windows OS:

 		set NEWVAR=SOMETHING

7-  Run the app from by running the following command: 
	
	flask run 

-- OR USE FOR MAC --

	python3 -m flask run

8-  Copy the URL from the terminal and run it in your browser.   // (Press CTRL+C to quit)

9-  Login with admin account and create other users ( admins and medical specialists)  or create a registered user account.

— The database is initialized with an admin account and 28 active diagnosis models. 

Entering the MongoDB Database: 

1- Download MongoDB Compass from https://www.mongodb.com/try/download/compass

2- After launching the app, connect to this connection string "mongodb+srv://graduationpojectai:W18UGYcRxlaBZY1X@cluster0.sbpeohk.mongodb.net/"

3- There will be several databases the one this website in using is GPDB 

4- You'll be able to view/edit all collections 

5- In case of any issue the MongoDB account used to create this cluster in mentioned below you can access it to toubleshoot any issues


-----------------------------------------------------------------------------------------
Accounts Information

MongoDB account credentials: 
Sign In Using Google Then: 
	- email: graduationprojectai@gmail.com
	- password: GraduationAI@24

Website admin account credentials:
	username : admin
	password: admin123


Note** this folder also contains a group meeting that describes how to edit each page feel free to see it in case you need to understand