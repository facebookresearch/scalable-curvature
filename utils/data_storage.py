import pandas as pd
import os

def get_rows_where_col_less(df, col, value):
    return df.loc[df[col] < value].copy()

class DataStorage:
    def __init__(self, columns):
        # data dictionary
        self.data = {col: [] for col in columns}
        self.columns = columns
        return

    def add_entry(self, entries):
        """
        Add one or more entries.
        
        Args:
            entries: Either a single dict or a list of dicts
                    e.g., {'step': 0, 'lr': 0.001, ...} 
                    or [{'step': 0, ...}, {'step': 0, ...}, ...]
        """
        # Handle single dict - convert to list
        if isinstance(entries, dict):
            entries = [entries]
        
        # Add all entries
        for entry in entries:
            for key, value in entry.items():
                if key in self.data:
                    self.data[key].append(value)
        return

    def save_to_csv(self, file_path):
        df = pd.DataFrame(self.data)
        df.to_csv(file_path, index=False)
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
