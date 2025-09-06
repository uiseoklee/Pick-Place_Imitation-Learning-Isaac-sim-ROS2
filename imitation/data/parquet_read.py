import pandas as pd
import numpy as np
import sys

def view_parquet():
    # pandas output settings - display full content without truncation
    pd.set_option('display.max_colwidth', None)  # prevent column content truncation
    pd.set_option('display.width', None)         # remove output width limit
    pd.set_option('display.max_rows', None)      # prevent row truncation (display all rows)
    np.set_printoptions(threshold=np.inf)        # prevent NumPy array truncation
    
    # read parquet file
    file_path = "sim_imitation_training_data.parquet"
    df = pd.read_parquet(file_path)
    
    # print basic information
    print(f"\nFile: {file_path}")
    print(f"Size: {df.shape[0]} rows x {df.shape[1]} columns")
    
    # display column list
    print("\nAvailable columns:")
    for i, col in enumerate(df.columns):
        print(f"{i}: {col}")
    
    # get user input
    try:
        print("\nEnter column numbers separated by commas (e.g., 0,1,3):")
        cols_input = input().strip()
        if cols_input.lower() == 'all':
            selected_cols = list(df.columns)
        else:
            col_indices = [int(x.strip()) for x in cols_input.split(',')]
            selected_cols = [df.columns[i] for i in col_indices]
        
        print("\nSelect how to specify row range:")
        print("1. View N rows from the beginning")
        print("2. View specific range of rows")
        print("3. View all rows")
        print("4. View rows conditionally")
        print("5. Check what data values are in a column")
        range_option = input().strip()
        
        if range_option == '1':
            print("Enter the number of rows to view:")
            rows = int(input().strip())
            print(df[selected_cols].head(rows))
        elif range_option == '2':
            print("Enter start row number (starting from 0):")
            start_row = int(input().strip())
            print("Enter end row number:")
            end_row = int(input().strip())
            # use slicing to select specific range of rows
            with pd.option_context('display.max_rows', None):
                print(df[selected_cols].iloc[start_row:end_row+1])  # +1 to include end_row
        elif range_option == '3':
            # display all rows (without truncation)
            with pd.option_context('display.max_rows', None):
                print(df[selected_cols])
        elif range_option == '4':
            print("Enter column number to apply condition:")
            filter_col_idx = int(input().strip())
            filter_col = df.columns[filter_col_idx]
            
            print(f"\nEnter condition to apply:")
            print("Supported operators: ==, >, <, >=, <=, !=")
            print("Examples: '== True' or '> 500' or '!= 0'")
            condition_input = input().strip()
            
            # parse condition
            try:
                # split condition (operator and value)
                if '>=' in condition_input:
                    op, value = '>=', condition_input.split('>=')[1].strip()
                elif '<=' in condition_input:
                    op, value = '<=', condition_input.split('<=')[1].strip()
                elif '==' in condition_input:
                    op, value = '==', condition_input.split('==')[1].strip()
                elif '!=' in condition_input:
                    op, value = '!=', condition_input.split('!=')[1].strip()
                elif '>' in condition_input:
                    op, value = '>', condition_input.split('>')[1].strip()
                elif '<' in condition_input:
                    op, value = '<', condition_input.split('<')[1].strip()
                else:
                    raise ValueError("Unsupported operator.")
                
                # try to convert value
                try:
                    # try to convert to number
                    numeric_value = float(value)
                    value = numeric_value
                except ValueError:
                    # if not a number, handle as string or boolean
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    # otherwise treat as string
                
                # filter according to condition
                if op == '==':
                    filtered_df = df[df[filter_col] == value]
                elif op == '!=':
                    filtered_df = df[df[filter_col] != value]
                elif op == '>':
                    filtered_df = df[df[filter_col] > value]
                elif op == '<':
                    filtered_df = df[df[filter_col] < value]
                elif op == '>=':
                    filtered_df = df[df[filter_col] >= value]
                elif op == '<=':
                    filtered_df = df[df[filter_col] <= value]
                
                print(f"\nRows that satisfy the condition '{condition_input}' in column '{filter_col}' (total {len(filtered_df)} rows):")
                print(filtered_df[selected_cols])
                
            except Exception as e:
                print(f"Error occurred while processing condition: {e}")
                print("Please enter a condition in the correct format.")
        elif range_option == '5':
            print("Enter column number to check data values:")
            col_idx = int(input().strip())
            if col_idx >= 0 and col_idx < len(df.columns):
                col_name = df.columns[col_idx]
                
                # check column data type
                col_type = df[col_name].dtype
                print(f"\nData type of column '{col_name}': {col_type}")
                
                # check unique value count
                unique_count = df[col_name].nunique()
                print(f"Number of unique values: {unique_count}")
                
                # check value distribution (array data is handled differently)
                if df[col_name].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
                    print("\nThis column contains array/list data.")
                    print("First 5 samples:")
                    for i, val in enumerate(df[col_name].head(5)):
                        print(f"  {i}: {val}")
                    
                    # check array length distribution
                    lengths = df[col_name].apply(lambda x: len(x) if hasattr(x, '__len__') else 1)
                    print("\nArray length distribution:")
                    print(lengths.value_counts().sort_index())
                else:
                    # check regular data using value_counts
                    print("\nValue distribution:")
                    value_counts = df[col_name].value_counts(dropna=False)
                    
                    # if too many values, display only top 20
                    if len(value_counts) > 20:
                        print("(Showing only top 20 values)")
                        for value, count in value_counts.head(20).items():
                            print(f"{value}: {count} items")
                        print(f"...and {len(value_counts)-20} more values.")
                    else:
                        for value, count in value_counts.items():
                            print(f"{value}: {count} items")
                    
                    # add statistical information for numeric data
                    if np.issubdtype(col_type, np.number):
                        print("\nBasic statistics:")
                        print(df[col_name].describe())
            else:
                print("Invalid column number.")
        else:
            print("Invalid input.")
            
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    view_parquet()