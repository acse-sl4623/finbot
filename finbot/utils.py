import re
"""
This module contains utility functions for various tasks.
Functions:
- is_alphabetical(s): Check if a string consists of alphabetical characters, spaces, and specific punctuation marks.
- find_alpha_column(df): Find the first column in a DataFrame where the majority of values are alphabetical.
- find_decimal_numbers(text): Find all decimal numbers in a given text.
- find_percentages(text): Find all percentages in a given text.
- find_quartiles(text): Find all quartiles (single digits between 1 and 4) in a given text.
- find_rank_within_sector(text): Find all ranks in the form of digit/digit in a given text.
- add_space_before_lines(input_string): Add a space before each line in a given string.

These are used in the pre-processing and knowledge base setup steps.
"""
# Tabe Preprocessing
def is_alphabetical(s):
    return all(c.isalpha() or c.isspace() or c in ("'", "’", ".", ",", '&','-','/') for c in s)

def find_alpha_column(df):
    for column in df.columns:
        if df[column].apply(lambda x: is_alphabetical(str(x))).mean() > 0.5:
            print(f"Found alphabetical column: {column}")
            return column
    return None

def find_decimal_numbers(text): # for allocation table
    # Regular expression pattern for finding decimal numbers
    pattern = r'[+-]?\d+\.\d+'
    
    # Find all matches in the provided text
    matches = re.findall(pattern, text)
    
    # Convert matches to float and return
    return [str(match) for match in matches]

def find_percentages(text): # for performance table
    # 1. Match strings starting with + or -, followed by digits, dot, and digits.
    # 2. Exclude those ending with a %.
    # 3. Match the exact string 'n/a'.
    performance_pattern = r'[+-]\d+\.\d+(?![\d%])|n/a'
    
    # Find all matches in the provided text
    matches = re.findall(performance_pattern, text)
    
    # Return matches as strings
    return [str(match) for match in matches]

def find_quartiles(text):
    # Use a regular expression to find all standalone single digits between 1 and 4
    single_digits_or_na = re.findall(r'\b([1-4]|n/a)\b', text)
    return single_digits_or_na

def find_rank_within_sector(text):
    # Use a regular expression to find all ranks of the form digit/digit
    ranks = re.findall(r'\d+\s*/\s*\d+|n/a', text)
    return ranks

# Knowledge base
def add_space_before_lines(input_string):
    # ChatGBT: https://chatgpt.com/share/67fb7af4-64cd-4fa5-86f1-f4a5a6566550
    # Split the string into a list of lines
    lines = input_string.split('\n')
    # Add a space before each line
    spaced_lines = [' ' + line for line in lines]
    # Join the lines back into a single string
    result = '\n'.join(spaced_lines)
    return result