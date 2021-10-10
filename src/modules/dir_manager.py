#! /usr/bin/env python

import os



class dir_manager:

    def __init__(self, path_dir):

        self.path_dir = path_dir

    def create_dir(self, path_dir):
        isPathAvailable = os.path.isdir(path_dir)
        if isPathAvailable == False:
            os.mkdir(path_dir)
