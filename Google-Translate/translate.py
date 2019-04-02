#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random, time
from mtranslate import translate


def main():

    infile=open('data/negative10k.txt', 'r')
    outfile=open('data/result/negative10kUZ.txt', 'a+')
    count=0;
    for line in infile.readlines():
	count+=1
	line=line.strip('\n')
	time.sleep(random.random()*3)
        translation=translate(line,'uz')
	print str(count) + ":" 
        print "En: " + line
        print "Uz: " + translation
	outfile.write(translation.encode('utf-8')+"\n")
    infile.close()
    outfile.close()

if __name__ == '__main__':
    main()
