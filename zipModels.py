'''
zip the candidate folder into zip archives.
'''
from zipfile import ZipFile
import os
import os.path as osp

# call in output older,  zip only folder with SLP
i = 0
for file in os.listdir('.'):
	# if i>0:     # a control for test
	# 	break
	if 'SLP' in file:
		svNm = file+'.zip'
		print('zipping', svNm)
		pthTar = osp.join(file, 'model_dump', 'checkpoint.pth')
		with ZipFile(svNm, 'w') as myzip:
			myzip.write(pthTar)
		i+=1

