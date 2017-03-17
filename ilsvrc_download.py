import os
import codecs
import urllib.request as url
import urllib

skip = 800
with codecs.open("ilsvrc11/fall11_urls.txt", "r",encoding='utf-8', errors='ignore') as text_file:
	#text_file = open("/home/dashmoment/data/fall11_urls.txt", "r")
	#lines = text_file.readlines()

	lines = text_file.read().split()

	for i in range(0, len(lines),2):
       
            try:
                if i>skip:
                    filename = "ilsvrc11/" + lines[i] + ".jpg"
                    url.urlretrieve(lines[i+1], filename)
                    print('{}'.format(lines[i+1]))
            except IOError as e:
                print(e)
                continue
		
            except UnicodeEncodeError as e:
                print(e)
                continue
              
	print('Finish download')
	text_file.close()
