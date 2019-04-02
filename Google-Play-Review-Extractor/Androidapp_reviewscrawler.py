#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A crawler to get reviews from specified apps.

I have created a list of top apps in file 'appsid'. 

After running the crawler: Two files are created positive.txt and negative.txt
"""

import re
import sys
import requests
#from lxml import html
#from pprint import pprint
import ast
#import json


def get_review(url):
    url = url+'&showAllReviews=true'
    print "### url: "+url
    try:
        response = requests.get(url, timeout=1.0)
	print response
    except requests.exceptions.Timeout as e:
        print('Connection Timeout')
        return
    except requests.exceptions.ConnectionError as e:
        print('No Internet connection')
        sys.exit(-1)
    #parse the body into tree
    htmlstring=response.text.encode('utf-8')
    cut_first=htmlstring.find("data:function(){return [[[\"gp:AOqpTO")
    if cut_first<=0:
	print "#############################33----------------NO REVIEWS FOUND IN THIS APP-------------------###############################"
	return
    htmlstring2=htmlstring[cut_first+23:]
    cut_last=htmlstring2.find("}});</script>")
    htmlstring3=htmlstring2[:cut_last]
    htmlstring4=htmlstring3.replace('null','0')
    htmlstring4=htmlstring4.replace('\n','')
    #htmlstring4=htmlstring4.replace("'","`")
    htmlstring4=htmlstring4.replace("true","1")
    htmlstring4=htmlstring4.replace("false","0")
    elementslist = ast.literal_eval(htmlstring4)

    count=0;
    for review in elementslist[0]:
	count+=1;
	star=review[2]
	comment=str(review[4]).replace('\n','')
	print "Review:"+str(count)+", Star:"+str(star)+", Comment:"+comment
	#check i fthe comment is not more than 250 chars long
	if len(comment) > 250:
		continue
	#write the star and the comment to reviews.txt file
	revs.write(str(star)+'\t'+comment+'\n')
	#write the comment to positive.txt file if the star>3
	if star > 3:
		pos.write(comment + '\n')
	#write the comment to negative.txt file if the star<=3
	if star <=3:
		neg.write(comment + '\n')
    
    
    return

try:
    with open('appsid2', 'r') as aid:
        urls = [i for i in aid.read().split('\n')]

except IOError:
    print('Error while opening App\'s ID file. make sure that\
    you have a file named "appsid" in the irectory of this\
    scrip and you have right permissions to access file. \nExiting...')

if __name__ == '__main__':
    #run it till there are ids in appsid file
    with open('reviews.txt', 'a+') as revs, open('positive.txt', 'a+') as pos, open('negative.txt', 'a+') as neg:
        length = len(urls)
        while length:
            try:
                for url in urls:
                    if len(url)>5:
			get_review(url)
                    print(str(length) + ' apps left')
                    length -= 1
            except IOError as e:
                print('Operation Failed Error...')
                pass

    print('Complete (Y)')
