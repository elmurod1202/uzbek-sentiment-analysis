#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A crawler to get reviews from specified apps.
I have created a list of top apps in file 'appsid'. This list is created
on a particular day of july month and it contains 3000 apps ids with their
category rank less than 10 on that specific day.
After running the crawler: Two files are created positive.txt and negative.txt
"""

import re
import sys
from lxml import html

def get_review(page):
    with open(page, 'r') as content_file:
	content = content_file.read()
    #parse the body into tree
    parsed = html.fromstring(content)
    #data extraction with xpath
    reviews = parsed.xpath('//*[@id="fcxH9b"]/div[4]/c-wiz/div/div[2]/div/div[1]/div/div/div[1]/div[2]/div/div')
    count=0;
    for element in reviews:
	count+=1
	#dastlabki 40 ta comment allaqachon narigi prog bilan olib bo'lingan, ulani yozmimiz
	if count<40:
		continue
	star=int(element.xpath('./div/div[2]/div[1]/div[1]/div/span[1]/div/div/@aria-label')[0][6])
	comment=' '.join(element.xpath('./div/div[2]/div[2]/span/text()'))
	comment=comment.encode('utf-8').replace('\n','')
	print "Review:"+str(count)+", Star:"+str(star)+", Comment:"+comment
	#check if the comment is not more than 250 chars long
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
    

    if not reviews:
	print "Could not retrieve reviews man"
        return
    return

if __name__ == '__main__':
    with open('reviews_big.txt', 'w') as revs, open('positive_big.txt', 'w') as pos, open('negative_big.txt', 'w') as neg:
	for file in range(1,15):
		print "######################################################-----------------------------------------------page:"+str(file)+"-----------------------------#####################"
		get_review("big_pages/big_page" + str(file))
    print('Complete (Y)')
