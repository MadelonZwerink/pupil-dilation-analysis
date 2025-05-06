# data.py

import re
from datamatrix import (
    DataMatrix,
    MultiDimensionalColumn,
    NAN)
from eyelinkparser import parse
import numpy as np

def read_data(data_path):
    # This function imports the data from asc files and saves it in the correct 
    # format. It creates new columns for the type of stimulus and the 
    # participant id.
    
    # The heavy lifting is done by eyelinkparser.parse()
    dm = parse(
        folder=data_path
    )
    
    # Extract only the column names (ignoring types)
    column_names = [col[0] if isinstance(col, tuple) else col for col in dm.columns]

    # Identify base variable names by removing the trial suffix
    variable_names = set()
    for col in column_names:
        parts = col.rsplit('_', 2)  # Split at the last underscore
        if len(parts) == 3 and parts[-1].isdigit():  # Check if suffix is a number (trial ID)
            variable_names.add(parts[0])  # Store the base variable name
        elif len(parts) == 4 and parts[-1].isdigit():
            variable_names.add(parts[0] + '_' + parts[1])  # Store the base variable name

    # Create a new cleaned DataMatrix
    cleaned_dm = DataMatrix(length=len(dm.path))
    cleaned_dm['path'] = dm['path']
    cleaned_dm['stimulus'] = dm['stimulus']
    cleaned_dm['trialid'] = dm['trialid']
    cleaned_dm['data_error'] = dm['data_error']

    # Process each row while handling multi-dimensional data
    for var in variable_names:
        data_count = 0
        cleaned_count = 0
        new_column = []
        max_depth = 1
        
        # Make sure to assign the correct shape based on the first column in dm
        
        for row in dm:
            trial = row['trialid']
            column_name = f"{var}_{trial}"
            if isinstance(row[column_name], np.ndarray):
                test_list = np.isnan(row[column_name])
                data_in_cell = sum(map(lambda i: not i, test_list)) 
                data_count += data_in_cell
               # if data_in_cell > 0:
                    #print("Column: " + str(column_name) + 
                    #      "\nTrial: " + str(trial))
            # If the column exists, copy the full content (including multi-dimensional values)
            if column_name in column_names:
                value = row[column_name]
                # Convert numpy arrays to lists
                if isinstance(value, np.ndarray):  
                    #print(value.shape)
                    if len(value) > max_depth:
                        max_depth = len(value)
                    new_column.append(value.tolist())  # Convert to list before adding
                elif isinstance(value, list):
                    new_column.append(value)  # Keep it as list
                else:
                    new_column.append(value)  # Add single value directly
            else:
                new_column.append([])  # Empty list for missing values
        
        if max_depth > 1:
            cleaned_dm[var] = MultiDimensionalColumn(shape=(max_depth,))
            for l in range(0,len(new_column)):
                cleaned_dm[var][l] = np.concatenate((np.asarray(new_column[l]), np.repeat(NAN,(max_depth - len(new_column[l])))))
                test_list_2 = np.isnan(cleaned_dm[var][l])
                cleaned_count += sum(map(lambda i: not i, test_list_2))     
            #print("The number of non-nan datapoints was: " + str(data_count))
            #print("Afterwards it is: " + str(cleaned_count))
        
        elif max_depth == 1:
            cleaned_dm[var] = new_column  # Assign extracted values to the new DataMatrix

    # Extract experimental conditions from column stimulus and store in stim_type
    emph_or_func = [word.split('_')[-1] for word in cleaned_dm['stimulus']]
    noise_or_fam = [word.split('_')[0] for word in cleaned_dm['stimulus']]

    stim_type = []
    for i in range(len(emph_or_func)):
        if noise_or_fam[i] == "noise":
            stim_type.append(emph_or_func[i] + "_noise")
        else:
            stim_type.append(emph_or_func[i])

    cleaned_dm.stim_type = stim_type

    # Extract participant labels from sessionid and store in participant
    cleaned_dm.sessionid = [re.search(r'\\00(\d+)_', filename).group(1) for filename in cleaned_dm.path]
    replace_dict = {12: 'inf2', 13: 'inf3', 19: 'inf4', 20: 'inf5'}
    cleaned_dm['participant'] = cleaned_dm['sessionid'] @ (lambda x: replace_dict.get(x, x))
    
    return cleaned_dm