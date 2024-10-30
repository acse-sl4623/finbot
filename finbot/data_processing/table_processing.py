import json
import os
import re
import pdfplumber
import pandas as pd
import PyPDF2
import tabula

# custom modules
import finbot.data_processing.test_tables as test_tables
import finbot.utils as utils

# 1. Processing pipeline for split Largest Holdings table
def extract_table_from_second_to_last_page(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        # Get the number of pages in the PDF
        total_pages = len(pdf.pages)
        
        # Get the second-to-last page
        second_to_last_page = pdf.pages[total_pages - 2]
        
        # Get the dimensions of the page
        width = second_to_last_page.width
        height = second_to_last_page.height
        
        # Define the bounding box for the bottom 30% of the page
        top_boundary = height * 0.8  # This is the 70% mark from the top
        bbox = (0, top_boundary, width, height)
        
        # Extract tables from the specified area
        cropped_page = second_to_last_page.within_bbox(bbox)
        tables = cropped_page.extract_tables()
        
        if not tables:
            return "No tables found in the bottom 20% of the second-to-last page."
        
        # Convert all tables to pandas DataFrames
        dataframes = [pd.DataFrame(table[1:], columns=table[0]) for table in tables]

        # Extract the DataFrame with 'Largest Holdings' in the column names
        largest_holdings_df = next(df for df in dataframes if 'Largest Holdings' in df.columns)
        
        return largest_holdings_df

def merge_tables(df1, df2, verbose=False):
    if df1.shape[1] != df2.shape[1]:
        print("Number of columns do not match")
        # Drop NaN columns first
        df1 = df1.dropna(axis=1, how='all')
        df2 = df2.dropna(axis=1, how='all')
        if df1.shape[1] != df2.shape[1]:
            print("Could not extract top issuers")
            return df2
    
    # Check if 'Rank' or 'Largest Holdings' is not in the columns of df2
    if 'Rank' not in df2.columns and 'Largest Holdings' not in df2.columns:
        # Turn column values into the first row values
        new_row = pd.DataFrame([df2.columns], columns=df2.columns)
        df2 = pd.concat([new_row, df2], ignore_index=True)

    col1 = utils.find_alpha_column(df1)
    col2 = utils.find_alpha_column(df2)

    if col1 and col2:
        # Reorder columns to put alphabetical columns first
        reordered_columns = [col1] + [col for col in df1.columns if col != col1]
        df1 = df1[reordered_columns]
        # print("Reordered columns in df1:")
        # display(df1)

        reordered_columns = [col2] + [col for col in df2.columns if col != col2]
        df2 = df2[reordered_columns]
        # print("Reordered columns in df2:")
        # display(df2)
        if verbose:
            print("df1 columns: ", df1.columns)
            print("df2 columns: ", df2.columns)
        # Rename columns in df2 to match df1 or vice versa based on 'Largest Holdings'
        if 'Largest Holdings' in df1.columns:
            df2.columns = df1.columns
        elif 'Largest Holdings' in df2.columns:
            df1.columns = df2.columns
    if verbose:
        print("Merged pdf has size ", len(df1.columns))
    merged_df = pd.concat([df1, df2], ignore_index=True)
    return merged_df

def create_merged_holdings_table_dict(pdf_file, verbose = False):
    #Function written with the help of ChatGBT (but share link could not be generated given sample pdf images were included)
    try:
        # Determine the number of pages in the PDF
        with open(pdf_file, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)
        
        if num_pages < 2 or num_pages > 5:
            raise ValueError(f"Unsupported number of pages: {num_pages}")

        # Extract tables from each page
        all_tables = []
        for page in range(1, num_pages + 1):
            tables = tabula.read_pdf(pdf_file, pages=page, stream=True, multiple_tables=True)
            if not isinstance(tables, list):
                print(f"Unexpected type for tables on page {page}: {type(tables)}")
                tables = []
            all_tables.extend(tables)

        # Check for split table condition and merge if needed
        if num_pages > 2 and all_tables[-1].iloc[0,0] != 1:
            print("Detected split table. Identifying top half.")
            if ('Largest Holdings' in all_tables[-2].columns) and ('Largest Holdings' in all_tables[-1].columns):
                print('Both parts of the table already extracted')
                df1 = all_tables[-2]
                df2 = all_tables[-1]
            else:
                print('Extracting top half of the split table')
                bottom_table = extract_table_from_second_to_last_page(pdf_file)
                if verbose:
                    print(bottom_table)
                df1 = bottom_table
                df2 = all_tables[-1]
                
            # Merge the two tables
            merged_df = merge_tables(df1, df2, verbose = verbose)
            if 'Rank' in merged_df.columns:
                merged_df = merged_df.drop(columns="Rank", axis=1)
            
            table_dict = {}
            for i in range(len(merged_df)):
                entry = {str(merged_df.iloc[i, 0]): str(merged_df.iloc[i, 1])}
                table_dict.update(entry)

        return table_dict
    
    except Exception as e:
        return None
    

# 2. TABULA extraction pipeline for perfromance tables
# 2a. Extracting tables
def extract_table_from_page(pdf_file, idx, page_nb, time_periods, verbose = False):
    #Function written with the help of ChatGBT (but share link could not be generated given sample pdf images were included)
    try:
        tables = tabula.read_pdf(pdf_file, pages=page_nb, stream=True, multiple_tables=True)
        if len(tables) < 1: # No tables found
            print(f"No tables found on page {page_nb}")
            if page_nb == 1:
                return None
            
            # It is possible that the discrete table and cumulative table are both on the first page
            # If that is the case we expect it to be the second table extracted hence idx + 1
            elif page_nb == 2:
                print("Trying page 1")
                return extract_table_from_page(pdf_file, idx + 1, 1, time_periods)
        else:
            # Check if any elements in time_periods appear in the DataFrame values or columns
            values_result = tables[idx].stack().map(lambda x: x in time_periods)
            columns_result = [col for col in tables[idx].columns if col in time_periods]

            any_time_periods_in_values = values_result.any().any()
            any_time_periods_in_columns = len(columns_result) > 0

            # Combine results
            any_time_periods = any_time_periods_in_values or any_time_periods_in_columns
            if not any_time_periods:
                print("Not the right table extracted as none of the time periods appear in the table.")
                return None
            else:
                return tables[idx]
    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")
        return None

# 2b. Pre-processing performance tables
def process_performance_df (df, time_periods):
    clean_dict = {}
    numerical_values = []
    #time_periods = ['3m', '6m', '1yr', '3yrs', '5yrs']

    # Store all numerical values in a list
    for index, row in df.iterrows():
        row_list = row[1:6].tolist()
        # check if the row values in columns 1 to 6 are not NaN
        if row[1:6].notna().all() and row_list!=time_periods:
            #print(row[1:6].tolist())
            numerical_values.append(row[1:6].tolist())

    #print(len(numerical_values))
    if len(numerical_values) == 4:
        clean_dict = {
            time_period: {
                'Fund': numerical_values[0][i],
                'Benchmark': numerical_values[1][i],
                'Rank within sector': numerical_values[2][i],
                'Quartile': numerical_values[3][i]
                }
            for i, time_period in enumerate(time_periods)
        }
    elif len(numerical_values) == 3:
        clean_dict = {
            time_period: {
                'Fund': numerical_values[0][i],
                'Rank within sector': numerical_values[1][i],
                'Quartile': numerical_values[2][i]
                }
            for i, time_period in enumerate(time_periods)
        }
    elif len(numerical_values) == 2:
        clean_dict = {
            time_period: {
                'Fund': numerical_values[0][i],
                'Benchmark': numerical_values[1][i],
                'Rank within sector': 'n/a',
                'Quartile': 'n/a'
                }
            for i, time_period in enumerate(time_periods)
        }
    return clean_dict

def extract_performance_dict_using_tabula(file, idx, page_nb, time_periods, pdf_directory = "factsheets/trustnet"):
    table = extract_table_from_page(os.path.join(pdf_directory, file), idx = idx, page_nb=page_nb, time_periods=time_periods)
    if table is not None:
        clean_dict = process_performance_df(table, time_periods)
        print(f"Performance dictionary for {file}: {clean_dict}")
        key_check = test_tables.check_perf_dict_keys(clean_dict, time_periods, file)
        value_check = test_tables.check_perf_dict_values(clean_dict, time_periods, file)
        if key_check and value_check:
            return clean_dict, key_check, value_check
        else:
            return None, key_check, value_check
    return None, None, None

# 3a. Extraction and processing pipeline for allocation tables using OCR
def extract_allocation_tables(json_file):
    output_file = "factsheets/json_files/all_allocation_tables.json"
    
    # Check if the output file already exists and load it
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping processing.")
        with open(output_file, 'r') as f:
            all_allocation_tables = json.load(f)
        return all_allocation_tables
    
    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Initialize a list to hold sector tables
    all_allocation_tables = {}
    idx = 1
    # Iterate through each filename in the loaded data
    for filename, elements in data.items():
        print(f"Processing file {filename} ({idx})")
        table_elements = elements.get("table_elements", [])
        all_allocation_tables[filename] ={}
        cnt = 0
        issuer_tables = 0
        for element in table_elements:
            text_content = element['text']  # Get the text from the element

            # Look for sector tables
            if 'Rank' in text_content and 'Sector' in text_content:
                all_allocation_tables[filename]["Sector Allocation"] = text_content
                #print("Sector table found")
                cnt += 1
            
            elif 'Rank' in text_content and 'Asset Classes' in text_content:
                all_allocation_tables[filename]["Asset Allocation"] = text_content
                #print("Asset table found")
                cnt += 1

            elif 'Rank' in text_content and 'Regions' in text_content:
                all_allocation_tables[filename]["Region Allocation"] = text_content
                print("Region table found")
                cnt += 1
            
            elif 'Rank' in text_content and 'Largest Holdings' in text_content:
                if issuer_tables == 0:
                    all_allocation_tables[filename]["Largest Holdings"] = text_content
                    print("Largest Holdings table found")
                    issuer_tables += 1
                else:
                    all_allocation_tables[filename]["Largest Holdings 2"] = text_content
                    print("Second Largest Holdings table found")
                cnt += 1
        idx += 1
    # Save all_tables to a JSON file
    output_file = "factsheets/json_files/all_allocation_tables.json"
    with open(output_file, 'w') as f:
        json.dump(all_allocation_tables, f, indent=4)
    return all_allocation_tables

# 3b. Extraction and processing pipeline for performance tables using OCR
def clean_performance_text(text, time_periods):
    # Define the starting time period
    time_period_start = time_periods[0]
    
    # Create a regex pattern to capture everything after the starting period
    relevant_pattern = fr'{re.escape(time_period_start)}\s*(.*)'
        
    match = re.search(relevant_pattern, text, re.DOTALL)  # Use DOTALL to match newlines
    
    if match:
        # Extract the text between the start and the Quartile match
        everything_after_start = match.group(1).strip()  # Remove leading/trailing whitespace
        print("Everything after time_period_start:", everything_after_start)
        return everything_after_start

    else:
        print("Performance pattern not found in text")
        print(text)
        return None

def convert_performance_to_dict(text, time_periods, filename, fund_type):
    # Find percentages
    percentages = utils.find_percentages(text)
    #print("BEFORE:Percentages: ", percentages)
    
    if len(percentages) % len(time_periods) == 1:
        print(f"{filename}: Mismatch between number of percentages and time periods")
        while len(percentages) % len(time_periods) != 0 and 'n/a' in percentages:
            percentages.remove('n/a')
    if len(percentages) < 1:
        print(f"{filename}: No percentages found")
        return None
    print("Percentages: ", percentages)
    
    if fund_type == 'relative':
        if len(percentages) // len(time_periods) == 2:
            fund_performance = percentages[:len(time_periods)]
            benchmark_performance = percentages[len(time_periods):]

            clean_dict = {
            time_period: {
                'Fund': fund_performance[i],
                'Benchmark': benchmark_performance[i],
                }
                for i, time_period in enumerate(time_periods)
            }
            #print("Clean dict: ", clean_dict)
            return clean_dict
        else:
            print(f"{filename} ({fund_type}): Mismatch between number of percentages and time periods")
            print(text)
            return None
        
    elif fund_type == 'absolute':
        # Absolute fund
        if len(percentages) == len(time_periods):
            fund_performance = percentages
            clean_dict = {
            time_period: {
                'Fund': fund_performance[i]
                }
                for i, time_period in enumerate(time_periods)
            }
            #print("Clean dict: ", clean_dict)
            return clean_dict
        else:
            print(f"{filename} ({fund_type}): Mismatch between number of percentages and time periods")
            print(text)
            return None
        
    elif fund_type == 'complete':
        # Get Rank within sector
        rank_within_sector = utils.find_rank_within_sector(text)
        if len(rank_within_sector) != len(time_periods):
            print(f"{filename} ({fund_type}): Mismatch between number of percentages and time periods")
            print("Ranks: ", rank_within_sector)
        elif len(rank_within_sector) < 1: # If no ranks are found, set to n/a
            print(f"{filename} ({fund_type}): No ranks found")
            rank_within_sector = ['n/a' for _ in range(len(time_periods))]
        
        # Get Quartiles
        quartiles = utils.find_quartiles(text)
        if len(quartiles) != len(time_periods):
            quartiles = quartiles[-5:]# Take the last 5 quartiles
            print(f"{filename} ({fund_type}): Mismatch between number of percentages and time periods")
            print("Quartiles: ", quartiles)
        elif len(quartiles) < 1: # If no quartiles are found, set to n/a
            print(f"{filename} ({fund_type}): No quartiles found")
            quartiles = ['n/a' for _ in range(len(time_periods))]
        
        # Build the dictionary
        if len(percentages) == len(time_periods):
            fund_performance = percentages
            clean_dict = {
            time_period: {
                'Fund': fund_performance[i],
                'Rank within sector': rank_within_sector[i],
                'Quartile': quartiles[i]
                }
                for i, time_period in enumerate(time_periods)
            }
            #print("Clean dict: ", clean_dict)
            return clean_dict
        elif len(percentages) // len(time_periods) == 2:
            fund_performance = percentages[:len(time_periods)]
            benchmark_performance = percentages[len(time_periods):]

            clean_dict = {
            time_period: {
                'Fund': fund_performance[i],
                'Benchmark': benchmark_performance[i],
                'Rank within sector': rank_within_sector[i],
                'Quartile': quartiles[i]
                }
                for i, time_period in enumerate(time_periods)
            }
            print("Clean dict: ", clean_dict)
            return clean_dict
        else:
            print(f"{filename} ({fund_type}): Mismatch between number of percentages and time periods")
            print(text)
            return None

def extract_performance_tables(json_file):
    output_file = "factsheets/json_files/all_perf_tables.json"
    output_stats = "factsheets/json_files/all_perf_stats.json"

    # Check if the output file already exists and load it
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping processing.")
        with open(output_file, 'r') as f:
            all_perf_tables = json.load(f)
    
    if os.path.exists(output_stats):
        print(f"Output file {output_stats} already exists. Skipping processing.")
        with open(output_stats, 'r') as f:
            complete_statistics = json.load(f)
        return all_perf_tables, complete_statistics
    
    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Load the outliers Excel file
    outliers_excel = "factsheets/outliers/Outlier_Structure_Funds.xlsx"
    outliers = pd.read_excel(outliers_excel)

    # Relative funds only have performance and benchmark returns
    relative_perf_funds = outliers[outliers['Nb of Rows'] == 2]
    relative_perf_funds_list = relative_perf_funds["SourceFile"].to_list()

    # Absolute funds have only one fund returns
    absolute_perf_funds = outliers[outliers['Nb of Rows'] == 1]
    absolute_perf_funds_list = absolute_perf_funds["SourceFile"].to_list()

    complete_perf_funds = outliers[outliers['Nb of Rows'] == 4]
    complete_perf_funds_list = complete_perf_funds["SourceFile"].to_list()

    # Initialize a list to hold sector tables
    all_perf_tables = {}
    file_idx = 1

    # Initialize a dictionary to hold extraction source statistics
    ext_source_stats = {'cperf': {'OCR' : {'count':0, 'files':[]}, 'tabula':{'count':0, 'files':[]}, 'extraction_errors': {'count':0, 'files':[]}},
                  'dperf': {'OCR' : {'count':0, 'files':[]}, 'tabula':{'count':0, 'files':[]}, 'extraction_errors': {'count':0, 'files':[]}}}
    
    # Initialize a dictionary to store types of files
    cperf_stats = {'relative': {'count': 0, 'files': [], 'errors': []}, 'absolute': {'count': 0, 'files': [], 'errors': []}, 'complete': {'count': 0, 'files': [], 'errors': []}}
    dperf_stats = {'relative': {'count': 0, 'files': [], 'errors': []}, 'absolute': {'count': 0, 'files': [], 'errors': []}, 'complete': {'count': 0, 'files': [], 'errors': []}}
    structure_outliers = {'structure_outliers': []} # these are the outliers whose performance tables are not on the first page (extracted with tabula)

    # Iterate through each filename in the loaded data
    for file, elements in data.items():
        print(f"Extracting for performance file {file} ({file_idx})")
        table_elements = elements.get("table_elements", [])
        all_perf_tables[file] ={}

        # We assume that the first page contains the cumulative performance table and the second page contains the discrete performance table
        cperf_pg_nb = 1
        dperf_pg_nb = 2

        # We assume both tables are on the same page
        cperf_idx = 0
        dperf_idx = 0

        if file in complete_perf_funds_list:
            print("This is an outlier file")
            cperf_pg_nb = int(complete_perf_funds.loc[complete_perf_funds['SourceFile'] == file, 'Cperf pg'])
            dperf_pg_nb = int(complete_perf_funds.loc[complete_perf_funds['SourceFile'] == file, 'Dperf pg'])

            if cperf_pg_nb == dperf_pg_nb:
                print("Performance tables are on the same page")
                cperf_idx = 0 # first table found should be cumulative performance
                dperf_idx = 1 # second table found should be discrete performance
                structure_outliers['structure_outliers'].append(file)

        # 1. Attempt extraction with tabula
        # a) Cumulative performance
        cperf_time_periods = ['3m', '6m', '1yr', '3yrs', '5yrs']
        cperf_dict, key_check, value_check = extract_performance_dict_using_tabula(file, idx = cperf_idx, page_nb = cperf_pg_nb, time_periods = cperf_time_periods)

        if cperf_dict is not None:
            all_perf_tables[file]["Cumulative Performance"] = cperf_dict
            ext_source_stats['cperf']['tabula']['count'] += 1
            ext_source_stats['cperf']['tabula']['files'].append(file)
            print('\n')

        else:
            print(f"Cumulative performance processing error for: {file}; key_check: {key_check}, value_check: {value_check}")
            print(f"Trying OCR for ...")

            # Define type of fund
            if file in relative_perf_funds_list:
                fund_type = 'relative'

            elif file in absolute_perf_funds_list:
                fund_type = 'absolute'

            else:
                fund_type = 'complete'

            # Extract table and text elements
            table_elements = elements.get("table_elements", [])
            text_elements = elements.get("text_elements", [])
            cperf_pattern = r'3m\s+6m\s+1yr\s+3yrs\s+5yrs'

            # Search in table elements
            for element in table_elements:
                text_content = element['text']
                if re.search(cperf_pattern, text_content):
                    # Clean the OCR performance extract
                    clean_cperf_text = clean_performance_text(text_content, cperf_time_periods)
                    if clean_cperf_text is None:
                        print(f"Failed to clean {file}")
                        cperf_stats[fund_type]['errors'].append(file)
                    else:
                        # Convert OCR performance extract to dictionary
                        cperf_dict = convert_performance_to_dict(clean_cperf_text, cperf_time_periods, file, fund_type)
                        if cperf_dict is None:
                            print(f"Failed to process {file}")
                            cperf_stats[fund_type]['errors'].append(file)
                        else:
                            # Check processing results gives expected format
                            key_test = test_tables.check_perf_dict_keys(cperf_dict, cperf_time_periods, file)
                            value_test = test_tables.check_perf_dict_values(cperf_dict, cperf_time_periods, file)
                            if key_test and value_test:
                                # Assign clean processed dictionary to all_perf_tables
                                all_perf_tables[file]["Cumulative Performance"] = cperf_dict
                                print("Cumulative performance extracted using OCR as table")
                                print("Clean dict: ", cperf_dict)
                                print('\n')
                                ext_source_stats['cperf']['OCR']['count'] += 1
                                ext_source_stats['cperf']['OCR']['files'].append(file)
                                cperf_stats[fund_type]['count'] += 1
                                cperf_stats[fund_type]['files'].append(file)

                            else:
                                print(f"Failed to process {file}; key_check: {key_check}, value_check: {value_check}")
                                print("Cumulative Performance dictionary: ", cperf_dict)
                                cperf_stats[fund_type]['errors'].append(file)
                                break

            # Search in text elements
            if 'Cumulative Performance' not in all_perf_tables[file]:
                for element in text_elements:
                    text_content = element['text']
                    if re.search(cperf_pattern, text_content):
                        # Clean the OCR performance extract
                        clean_cperf_text = clean_performance_text(text_content, cperf_time_periods)
                        if clean_cperf_text is None:
                            print(f"Failed to clean {file}")
                            cperf_stats[fund_type]['errors'].append(file)
                        else:
                            # Convert OCR performance extract to dictionary
                            cperf_dict = convert_performance_to_dict(clean_cperf_text, cperf_time_periods, file, fund_type)
                            if cperf_dict is None:
                                print(f"Failed to process {file}")
                                cperf_stats[fund_type]['errors'].append(file)
                            else:
                                # Check processing results gives expected format
                                key_test = test_tables.check_perf_dict_keys(cperf_dict, cperf_time_periods, file)
                                value_test = test_tables.check_perf_dict_values(cperf_dict, cperf_time_periods, file)
                                if key_test and value_test:
                                    # Assign clean processed dictionary to all_perf_tables
                                    all_perf_tables[file]["Cumulative Performance"] = cperf_dict
                                    print("Cumulative performance extracted using OCR as table")
                                    print("Clean dict: ", cperf_dict)
                                    print('\n')
                                    ext_source_stats['cperf']['OCR']['count'] += 1
                                    ext_source_stats['cperf']['OCR']['files'].append(file)
                                    cperf_stats[fund_type]['count'] += 1
                                    cperf_stats[fund_type]['files'].append(file)

                                else:
                                    print(f"Failed to process {file}; key_check: {key_check}, value_check: {value_check}")
                                    print("Cumulative Performance dictionary: ", cperf_dict)
                                    cperf_stats[fund_type]['errors'].append(file)
                                    break

            # Store files for which extraction was not possible
            if 'Cumulative Performance' not in all_perf_tables[file]:
                print(f"No cumulative performance table found for {file}")
                ext_source_stats['cperf']['extraction_errors']['count'] += 1
                ext_source_stats['cperf']['extraction_errors']['files'].append(file)
                print('\n')
        
        # b) Discrete performance
        dperf_time_periods = ['0-12m', '12m-24m', '24m-36m', '36m-48m', '48m-60m']
        dperf_dict, key_check, value_check = extract_performance_dict_using_tabula(file, dperf_idx, dperf_pg_nb, dperf_time_periods)
        if dperf_dict is not None:
            all_perf_tables[file]["Discrete Performance"] = dperf_dict
            ext_source_stats['dperf']['tabula']['count'] += 1
            ext_source_stats['dperf']['tabula']['files'].append(file)
            print('\n')
        else:
            print(f"Discrete performance processing error for: {file}; key_check: {key_check}, value_check: {value_check}")
            print(f"Trying OCR for ...")
            # Discrete performance only available as Text element
            text_elements = elements.get("text_elements", [])
            dperf_pattern = r'0-12m\s+12m-24m\s+24m-36m\s+36m-48m\s+48m-60m'

            for element in text_elements:
                text_content = element['text']
                if re.search(dperf_pattern, text_content):
                    # Clean the OCR performance extract
                    clean_dperf_text = clean_performance_text(text_content, dperf_time_periods)
                    if clean_dperf_text is None:
                        print(f"Failed to clean {file}")
                        dperf_stats[fund_type]['errors'].append(file)
                    else:

                        # Convert OCR performance extract to dictionary
                        dperf_dict = convert_performance_to_dict(clean_dperf_text, dperf_time_periods, file, fund_type)
                        if dperf_dict is None:
                            print(f"Failed to process {file}")
                            dperf_stats[fund_type]['errors'].append(file)
                        else:
                            # Check processing results give expected format
                            key_test = test_tables.check_perf_dict_keys(dperf_dict, dperf_time_periods, file)
                            value_test = test_tables.check_perf_dict_values(dperf_dict, dperf_time_periods, file)
                            if key_test and value_test:
                                # Assign clean processed dictionary to all_perf_tables
                                all_perf_tables[file]["Discrete Performance"] = dperf_dict
                                print("Discrete Performance extracted using OCR as text")
                                print("Discrete Performance dictionary: ", dperf_dict)
                                print('\n')
                                ext_source_stats['dperf']['OCR']['count'] += 1
                                ext_source_stats['dperf']['OCR']['files'].append(file)
                                dperf_stats[fund_type]['count'] += 1
                                dperf_stats[fund_type]['files'].append(file)

                            else:
                                print(f"Failed to process {file}; key_check: {key_check}, value_check: {value_check}")
                                print("Discrete Performance dictionary: ", dperf_dict)
                                dperf_stats[fund_type]['errors'].append(file)
                                break
            
            # Store files for which extraction was not possible
            if 'Discrete Performance' not in all_perf_tables[file]:  
                print(f"No discrete performance table found for {file}")
                ext_source_stats['dperf']['extraction_errors']['count'] += 1
                ext_source_stats['dperf']['extraction_errors']['files'].append(file)
        file_idx += 1
        print('----------------------------------------------------')
    # Save all_tables to a JSON file
    output_file = "factsheets/json_files/all_perf_tables.json"
    with open(output_file, 'w') as f:
        json.dump(all_perf_tables, f, indent=4)

    complete_statistics = {
        'ext_source_stats': ext_source_stats,
        'cperf': cperf_stats,
        'dperf': dperf_stats,
        'structure_outliers': structure_outliers
    }

    output_file = "factsheets/json_files/all_perf_stats.json"
    with open(output_file, 'w') as f:
        json.dump(complete_statistics, f, indent=4)
    
    return all_perf_tables, complete_statistics

# 4. Preprocessing pipeline
# 4a. Preprocessing performance allocation tables
def convert_allocation_df_to_dict(text):
    # Initialize a dictionary to hold the structured data
    names = []
    percentages = utils.find_decimal_numbers(text)
    # Clean the string and split into parts
    parts = text.split()

    # Initialize an index for tracking position in the parts list
    i = 0

    # Loop through parts to extract sector names and percentages
    while i < len(parts):
        # Check if the current part is a digit (indicating a rank)
        #print(f"i = {i} and parts[i] = {parts[i]}")
        if parts[i].isdigit(): # everything after Rank
            # Move to the next part to get the sector name
            i += 1
            name = parts[i]  # Start with the next part as the sector name
            #print(f"i = {i} and parts[i] = {parts[i]}")

            # If the sector name consists of more than one word, join them
            while (i + 1 < len(parts) and # does not exceed text length
                    not parts[i + 1].isdigit() and # is not a rank
                    parts[i + 1] not in percentages): # is not a percentage
                
                name += " " + parts[i + 1]
                i += 1
            names.append(name)
            #print("Name: ", name)

        # Move to the next part
        i += 1
    print("Names: " , names)
    print("Percentages: ", percentages)
    if len(names) != len(percentages):
        print("Mismatch between number of names and percentages")
        return None
    clean_dict = {sector: percentage for sector, percentage in zip(names, percentages)}
    print("Clean dict: ", clean_dict)
    return clean_dict

def preprocess_allocation_tables(all_tables_str, pdf_directory = "factsheets/trustnet"):
    output_file = "factsheets/json_files/all_allocation_dicts.json"
    
    # Check if the output file already exists and load it
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping processing.")
        with open(output_file, 'r') as f:
            all_allocation_tables_dict = json.load(f)
        return all_allocation_tables_dict
    
    all_allocation_tables_dict = {}
    for idx_doc, filename in enumerate(os.listdir(pdf_directory)):
        if filename.endswith('.pdf') and filename in all_tables_str.keys():
            all_allocation_tables_dict[filename] = {}
            print(f"Processing {filename}: {idx_doc + 1}/{len(os.listdir(pdf_directory))}")
            for key, value in all_tables_str[filename].items():
                clean_dict = convert_allocation_df_to_dict(value)
                if clean_dict is not None:
                    all_allocation_tables_dict[filename][key] = clean_dict
                else:
                    print(f"Could not process {key} for {filename}")
                print("\n")
            
            # Check if there are two 'Largest Holdings' tables already identified through OCR and merge them
            if 'Largest Holdings' in all_allocation_tables_dict[filename].keys() and 'Largest Holdings 2' in all_allocation_tables_dict[filename].keys():
                # Merge the two dictionaries:
                print("Merging two largest holdings tables")
                all_allocation_tables_dict[filename]['Largest Holdings'] = {**all_allocation_tables_dict[filename]['Largest Holdings'], **all_allocation_tables_dict[filename]['Largest Holdings 2']}
                del all_allocation_tables_dict[filename]['Largest Holdings 2']

            # If only one 'Largest Holdings' table is identified, check if it has less than 3 entries and extract the second table
            elif 'Largest Holdings' in all_allocation_tables_dict[filename].keys():
                if len(all_allocation_tables_dict[filename]["Largest Holdings"].keys())<3:
                    pdf_file = os.path.join(pdf_directory, filename)
                    all_allocation_tables_dict[filename]["Largest Holdings"] = create_merged_holdings_table_dict(pdf_file, verbose = True)
                    print("Largest holdings: ", all_allocation_tables_dict[filename]["Largest Holdings"])
            print("----------------------------------------------------")
        
        # Save all_tables to a JSON file
        output_file = "factsheets/json_files/all_allocation_dicts.json"
        with open(output_file, 'w') as f:
            json.dump(all_allocation_tables_dict, f, indent=4)
    return all_allocation_tables_dict

# 5. Formating the dictionaries
def format_entries_for_embedding(tables, verbose = False):
    full_text = ''
    for table_header, table in tables.items(): # tables is a dictionary of dictionaries
        full_text = full_text + table_header + '\n'
        print(table_header)
        print(f"Table associated to key {table_header} : {tables[table_header]}")
        for element_key, element_value in table.items():
            if isinstance(element_value, dict):
                time_period = element_key
                time_period_values = element_value
                full_text += "Time period " + time_period + ': '
                full_text += ", ".join([": ".join([str(k), str(v)]) for k, v in time_period_values.items()])
                full_text += '\n'
            else:
                full_text += ": ".join([str(element_key), str(element_value)]) + '\n'
        full_text += '\n'  # Double newline to separate sections
    if verbose:
        print(full_text)
    return full_text