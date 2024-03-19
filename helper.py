# -*- coding: utf-8 -*-
# @Date  : 03.01.24
# @Author  : Hark Karkee
# @Email   : hkarkee@iastate.edu
# @File    : helper.py
# @Software: PyCharm

import os
import glob

def remove_files():
    if os.path.isdir("data"):
        files = glob.glob('data/*')
        for f in files:
            # pass
            os.remove(f)
    else:
        os.mkdir("data")

    return 1

def savefileindrive(uploaded_file):
    try:
        print(os.path.dirname(os.path.realpath(__file__)))
        if remove_files() ==1:
            with open(os.path.join("data", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        return 1

    except Exception as e:
        print(e)