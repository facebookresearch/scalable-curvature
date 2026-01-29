# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import os

def get_rows_where_col_less(df, col, value):
    return df.loc[df[col] < value].copy()

class DataStorage:
    def __init__(self, columns, num_params, embd_params):
        # data dictionary
        self.data = {col: [] for col in columns}

        self.columns = columns
        self.num_params = num_params
        self.embd_params = embd_params
        return

    def add_entry(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.data:
                self.data[key].append(value)
        return

    def save_to_csv(self, file_path):

        df = pd.DataFrame(self.data)
        df['num_params'] = self.num_params
        df['embd_params'] = self.embd_params

        df.to_csv(file_path, index = False)
        return
    
    def load_from_csv(self, file_path, num_steps):
        " Returns True if the file was loaded successfully, False if the file does not exist. "

        if os.path.exists(file_path):

            df = pd.read_csv(file_path)
            df = get_rows_where_col_less(df, 'step', num_steps)
            
            # Extract num_params and embd_params if they exist in the file
            if 'num_params' in df.columns:
                df = df.drop(columns=['num_params'])
            
            if 'embd_params' in df.columns:
                df = df.drop(columns=['embd_params'])
            
            # Update the data dictionary
            for col in self.data.keys():
                if col in df.columns:
                    self.data[col] = df[col].tolist()
                else:
                    self.data[col] = []
            return True
        return False


class DataStorageGeneral:
    def __init__(self, columns):
        # data dictionary
        self.data = {col: [] for col in columns}

        self.columns = columns
        return

    def add_entry(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.data:
                self.data[key].append(value)
        return

    def save_to_csv(self, file_path):

        df = pd.DataFrame(self.data)
        df.to_csv(file_path, index = False)
        return
    
    def load_from_csv(self, file_path, num_steps):
        " Returns True if the file was loaded successfully, False if the file does not exist. "

        if os.path.exists(file_path):

            df = pd.read_csv(file_path)
            df = get_rows_where_col_less(df, 'step', num_steps)
            
            # Update the data dictionary
            for col in self.data.keys():
                if col in df.columns:
                    self.data[col] = df[col].tolist()
                else:
                    self.data[col] = []
            return True
        return False

        