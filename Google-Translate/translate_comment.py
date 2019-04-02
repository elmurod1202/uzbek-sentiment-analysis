#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random, time
from mtranslate import translate
from difflib import SequenceMatcher

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def main():

    infile=open('data/reviews_to_translate.txt', 'r')
    outfile=open('data/result/reviews_translated.txt', 'a+')
    outfile2=open('data/result/translation_results.txt', 'a+')
    count=0;
    for line in infile.readlines():
	count+=1
	line=line.strip('\n')
	time.sleep(random.random()*1)
        star=line[:2]
	sentence=line[2:]
	translation=translate(sentence,'uz')
	sml=similarity(sentence, translation)
	print str(count) + ": " + str(sml) 
        print ">>> " + sentence
        print "<<< " + translation
	
	#newline=""	
	if sml>0.7:
		newline = star+sentence
	else:
		newline = star+translation
		outfile2.write(">>> " + sentence + "\n")
		outfile2.write("<<< " + translation + "\n\n")
	outfile.write(newline+"\n")
    infile.close()
    outfile.close()
    outfile2.close()

if __name__ == '__main__':
    main()
