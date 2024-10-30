from langchain_community.document_loaders  import PyPDFLoader
import sys
import os
import re

import finbot.data_processing.table_processing as tp
import finbot.data_processing.fund_info_processing as fip
import finbot.data_extraction.table_extraction_stats as tes

def pdf_file_processing_pipeline(pdf_file, verbose = False):
    OCR_output = "factsheets/json_files/all_elements_by_file.json"
    all_allocation_tables = tp.extract_allocation_tables(OCR_output) # extract allocation tables
    all_allocation_dicts = tp.preprocess_allocation_tables(all_allocation_tables) # preprocess allocation tables
    all_performance_tables, complete_stats = tp.extract_performance_tables(OCR_output) # extract performance tables

    # Plot extraction statistics
    #tes.compute_statistics(all_allocation_tables)
    #tes.plot_performance_statistics(complete_stats['ext_source_stats'])
    try:
        loader = PyPDFLoader(pdf_file)
        pdf = loader.load()
        information = {'fund': {}, 'tables': {}, 'missing': {'fund': [], 'tables': []}}
        # 1. Extract only the relevant text from the first page
        first_page = pdf[0].page_content
        extracted_information = fip.process_fund_information(first_page, pdf_file)
        if extracted_information['metadata']['inv_obj'] == 'No investment objective provide for this fund':
            print("Asking LLM for investment objective")
            extracted_information['metadata']['inv_obj'] = fip.ask_llm_for_investment_objective(first_page)
        print("Extracted fund information")
        fund_metadata = extracted_information['metadata']
        fund_information = extracted_information['summary']

        # Store fund information in the information dictionary
        information['fund']['metadata'] = fund_metadata
        information['fund']['text'] = fund_information

        # 2. Extract the table data
        file_name = re.search(r'[^/]+\.pdf', pdf_file).group()
        if file_name not in all_performance_tables.keys():
            information['missing']['tables'].append(file_name)
        elif file_name not in all_allocation_dicts.keys():
            information['missing']['fund'].append(file_name)
        else:
            allocation_tables = all_allocation_dicts[file_name]
            performance_tables = all_performance_tables[file_name]
            tables_metadata = {**allocation_tables, **performance_tables}
            print("Extracted table metadata")
            print(type(tables_metadata))
            tables_information = tp.format_entries_for_embedding(tables_metadata, verbose = verbose)
            print("Extracted table information")

            # Store the tables in the information dictionary
            information['tables']['metadata'] = tables_metadata
            information['tables']['text'] = tables_information
    except Exception as e:
        print(f"Error processing {pdf_file}")
        print(e)
        return None
    return information