#import cv2
#import numpy as np
from SciServer import CasJobs, Authentication, Files
#import base64
import os
import pandas
import inotify.adapters

#This script is the same as FrameUploader except it uploads the image to
#a volume file storage rather than storing inside Casjobs
#Instead it stores a path to the image in CasJobs

CLOSE_WRITE = "IN_CLOSE_WRITE"
MOVED_TO = "IN_MOVED_TO"
class FrameFileUploader:

	def login(self):
		self.authToken = Authentication.login()

	def insertImage(self, path, filename):
		#encoded_string = ''
		#with open(filename, "rb") as image_file:
    		#	encoded_string = base64.b64encode(image_file.read())		
		name=os.path.basename(filename)
		sciserverFramePath = os.path.join(self.remoteFrameDir, filename)
		self.uploadFile(localPath=os.path.join(path, filename), remotePath=sciserverFramePath)
		
		my_columns = ['Id', 'Name', 'Path', 'Processed']
		values = [[0, name, sciserverFramePath, 0]]

		df_to_insert = pandas.DataFrame(data=values, index=[0], columns=my_columns)
		CasJobs.uploadPandasDataFrameToTable(dataFrame=df_to_insert, tableName='FramePathProcessing', context=self.my_context)
		



	def createImageTable(self):
		create_table_query = '''
				CREATE TABLE FramePathProcessing
				(
				Id int NOT NULL IDENTITY PRIMARY KEY,
				Name varchar(50),
				Path varchar(150),
				Processed int NOT NULL,
                goodMeltScore float,
                fastBottomScore float,
                fastTopScore float,
                processedPath varchar(150),
                cropPath varchar(150)
				)'''

		CasJobs.executeQuery(sql=create_table_query, context=self.my_context)

	def dropImageTable(self):
		drop_table_query = "DROP TABLE FramePathProcessing"
		CasJobs.executeQuery(sql=drop_table_query, context=self.my_context)

	def beginWatch(self):
		for event in self.inot.event_gen(yield_nones=False):
			(_, type_names, path, filename) = event
			for type_name in type_names:
				print("PATH=[{}] FILENAME=[{}] EVENT_TYPES={}".format(path, filename, type_names))
				if type_name == CLOSE_WRITE or type_name == MOVED_TO:
					print("beginning insert")
					try:
                        self.insertImage(path, filename)
                    except FileNotFoundError:
                        print("File not found, continuing")
					#move file to "sent" directory
					
					#os.rename(os.path.join(self.dir_to_watch, filename), os.path.join(self.dir_to_watch, "inserted_images", filename))
					#os.remove(os.path.join(self.dir_to_watch, filename)) #Tyrell takes care of this...

	def uploadFile(self, localPath, remotePath):
                Files.upload(fileService=self.fs, path=remotePath, localFilePath=localPath)

	def __init__(self, hot_dir, my_context="MyDB"):
		
		self.login()
		self.dir_to_watch = hot_dir
		self.my_context = "MyDB"
		self.inot = inotify.adapters.Inotify()
		print(self.dir_to_watch)
		self.inot.add_watch(self.dir_to_watch)
		
		self.fs = Files.getFileServices()[0]
		self.remoteFrameDir = 'Temporary/ncarey/scratch/Frames'
                

if __name__ == "__main__":

	hot_dir = os.path.join("/live", "vnc-TiltLDFZ")
	frame_man = FrameFileUploader(hot_dir)
	frame_man.beginWatch()
