import cv2
import numpy as np
from SciServer import CasJobs, Authentication
import base64
import os
import pandas
import inotify.adapters

CLOSE_WRITE = "IN_CLOSE_WRITE"
MOVED_TO = "IN_MOVED_TO"
class FrameUploader:

	def login(self):
		self.authToken = Authentication.login()

	def insertImage(self, filename):
		encoded_string = ''
		with open(filename, "rb") as image_file:
    			encoded_string = base64.b64encode(image_file.read())		
		name=os.path.basename(filename)
		
		my_columns = ['Id', 'Name', 'Photo']
		values = [[0, name, encoded_string]]

		df_to_insert = pandas.DataFrame(data=values, index=[0], columns=my_columns)
		CasJobs.uploadPandasDataFrameToTable(dataFrame=df_to_insert, tableName='ImageTableDemo', context=self.my_context)
		



	def createImageTable(self):
		create_table_query = '''
				CREATE TABLE ImageTableDemo
				(
				Id int NOT NULL IDENTITY PRIMARY KEY,
				Name varchar(50),
				Photo varchar(max)
				)'''

		CasJobs.executeQuery(sql=create_table_query, context=self.my_context)

	def dropImageTable(self):
		drop_table_query = "DROP TABLE ImageTableDemo"
		CasJobs.executeQuery(sql=drop_table_query, context=self.my_context)

	def beginWatch(self):
		for event in self.inot.event_gen(yield_nones=False):
			(_, type_names, path, filename) = event
			for type_name in type_names:
				print("PATH=[{}] FILENAME=[{}] EVENT_TYPES={}".format(path, filename, type_names))
				if type_name == CLOSE_WRITE or type_name == MOVED_TO:
					print("beginning insert")
					self.insertImage(filename)
					#move file to "sent" directory
					
					#os.rename(os.path.join(self.dir_to_watch, filename), os.path.join(self.dir_to_watch, "inserted_images", filename))
					os.remove(os.path.join(self.dir_to_watch, filename))

	def __init__(self, hot_dir, my_context="MyDB"):
		
		self.login()
		self.dir_to_watch = hot_dir
		self.my_context = "MyDB"
		self.inot = inotify.adapters.Inotify()
		print(self.dir_to_watch)
		self.inot.add_watch(self.dir_to_watch)


if __name__ == "__main__":

	hot_dir = os.path.join("/live", "vnc-TiltLDFZ")
	frame_man = FrameUploader(hot_dir)
	frame_man.beginWatch()
