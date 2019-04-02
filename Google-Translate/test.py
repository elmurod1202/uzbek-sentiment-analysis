#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mtranslate import translate


def main():
    to_translate = '1	A Very bad app'
    print(to_translate)
    print(translate(to_translate, 'uz'))

if __name__ == '__main__':
    main()
