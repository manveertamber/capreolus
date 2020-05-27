#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Crystina Zhang time:26/5/2020


def add(a, b):
    return a + b


def is_palindrome(s):
    """ check whether the string is a palindrome"""
    return s == s[::-1]

def long_palindrome_index(s):
    """ return list index of elements of palindrome and at least six characters"""
    
