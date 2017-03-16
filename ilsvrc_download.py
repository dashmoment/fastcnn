import os
import codecs
import urllib.request as url
import urllib


with codecs.open("/home/dashmoment/data/urllists/fall11_urls.txt", "r",encoding='utf-8', errors='ignore') as text_file:
	#text_file = open("/home/dashmoment/data/fall11_urls.txt", "r")
	#lines = text_file.readlines()

	lines = text_file.read().split()

	for i in range(0, len(lines),2):

		try:

			filename = "/home/dashmoment/data/" + lines[i] + ".jpg"
			url.urlretrieve(lines[i+1], filename)
			print('{}'.format(lines[i+1]))
		except urllib.error.URLError as e:
			print(e)
			continue

	print('Finish download')
	text_file.close()

