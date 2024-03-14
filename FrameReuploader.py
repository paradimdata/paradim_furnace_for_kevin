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

class FrameReuploader:


	def insertImage(self, path, filename):
		#encoded_string = ''
		#with open(filename, "rb") as image_file:
    		#	encoded_string = base64.b64encode(image_file.read())		
		name=os.path.basename(filename)
		sciserverFramePath = os.path.join(self.remoteFrameDir, filename)
		self.uploadFile(localPath=os.path.join(path, filename), remotePath=sciserverFramePath)
		
		


		


	def uploadDir(self):
		for frameFile in os.listdir(self.dir_to_insert):
            
            file_path = os.path.join(self.dir_to_insert, frameFile)
            my_columns = ['Id', 'Name', 'Path', 'Processed', 'goodMeltScore', 'fastBottomScore', 'fastTopScore', 'processedPath']
            values = [[0, frameFile, file_path, 0]]
            df_to_insert = pandas.DataFrame(data=values, index=[0], columns=my_columns)
            CasJobs.uploadPandasDataFrameToTable(dataFrame=df_to_insert, tableName='FramePathProcessing1', context=self.my_context)

	def __init__(self, dir_to_insert, my_context="MyDB"):
		

		self.dir_to_insert = dir_to_insert
		self.my_context = "MyDB"
                

if __name__ == "__main__":

	frame_dir = ""
	frame_man = FrameReuploader(hot_dir)
	frame_man.uploadDir()
