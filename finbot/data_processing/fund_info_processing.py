import json
import re
from langchain_community.llms import Ollama

import finbot.data_processing.test_info as test_info

# 1.a. Fund Information Extraction and Pre-processing (with LLM)
def extract_generated_text(response):
    generated_text = ""
    if hasattr(response, 'generations'):
        generations = response.generations
        for generation_chunk in generations:
            for chunk in generation_chunk:
                if hasattr(chunk, 'text'):
                    generated_text += chunk.text
    else:
        print("The response object does not have a 'generations' attribute.")
    print("Generated Text:", generated_text)
    return generated_text

def clean_generated_text(generated_text):
    # Extract the JSON part using regex
    json_str = re.search(r'```json(.*?)```', generated_text, re.DOTALL).group(1).strip()

    # Remove trailing commas (if there were any)
    cleaned_text = re.sub(r',\s*([\]}])', r'\1', json_str)
    return cleaned_text

def parse_json(cleaned_text):
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return {"error": "Failed to decode JSON from response"}

def ask_llm_for_investment_objective(text, llm=Ollama(model="gemma:2b", temperature=0.0)):
    # ChatGBT (combining prompts): https://chatgpt.com/share/fd36c607-f28b-40ad-a3ad-1de6e91495ed
    # Define the prompt for summarizing the fund information
    summary_prompt = (
        f"Extract the investment objective from the following text:\n"
        f"{text}\n\n"
        f"Return only the summary text directly."
    )

    # Generate responses for both prompts
    summary_response = llm.generate([summary_prompt])

    # Extract and clean the generated text for the summary
    summary_text = extract_generated_text(summary_response)
    cleaned_summary_text = summary_text.split('\n', 1)[-1]

    return {"summary": cleaned_summary_text}

# 1.b. Function to extract fund information metadata (NO LLM)
def extract_fund_info_metadata(text):
    try:
        print("First page split by performance")
        first_half = text.split('Cumulative Performance')[0]
        second_half = text.split('Cumulative Performance')[1]
    except IndexError:
        print("First page split by fund information")
        first_half = text.split('Fund Information')[0]
        second_half = text.split('Fund Information')[1]

    # Define the regex patterns
    first_half_patterns = {
        'fund_name': r'2024(.*?)What is the Fund’s objective\?',
        'inv_obj': r'What is the Fund’s objective\?(.*)'
    }

    second_half_patterns = {
        'asset_class': r'Asset Class\s*(.*?)\s*Launch Date',
        'launch_date': r'Launch Date\s*(.*?)\s*Fund Size',
        'SEDOL': r'SEDOL\s*(.*?)\s*ISIN',
        'ISIN': r'ISIN\s*(.*?)\s*Data provided by FE fundinfo'
    }
    
    info = {}
    
    for key, pattern in first_half_patterns.items():
        match = re.search(pattern, first_half, re.DOTALL)
        if match:
            extracted_text = match.group(1).strip()
            info[key] = extracted_text
        else:
            print(f"Pattern not found for {key}.")
            info[key] = 'N/A'
    
    for key, pattern in second_half_patterns.items():
        match = re.search(pattern, second_half, re.DOTALL)
        if match:
            extracted_text = match.group(1).strip()
            info[key] = extracted_text
        else:
            print(f"Pattern not found for {key}.")
            info[key] = 'N/A'
    return info

def generate_fund_summary(fund_metadata):
    fund_summary = ''
    fund_summary += f"The fund {fund_metadata['fund_name'] } is a {fund_metadata['asset_class']} fund. \n"
    fund_summary += f"{fund_metadata['inv_obj']}\n"
    fund_summary += f"The fund was launched on {fund_metadata['launch_date']}.\n"
    if fund_metadata['SEDOL'] != 'N/A' or fund_metadata['ISIN'] != 'N/A':
        fund_summary += f"The fund has the following identifiers: \n"
        if fund_metadata['SEDOL'] != 'N/A':
            fund_summary += f"SEDOL: {fund_metadata['SEDOL']} \n"
        if fund_metadata['ISIN'] != 'N/A':
            fund_summary += f"ISIN: {fund_metadata['ISIN']} \n"
    return fund_summary

def process_fund_information(text, filename, verbose = False):
    # Extract the content until the 'Cumulative Performance' section
    fund_metadata = extract_fund_info_metadata(text)
    fund_metadata = test_info.check_SEDOL_ISIN_launch_date(fund_metadata, filename)
        
    if verbose == True:
        print(f"Fund metadata: {fund_metadata.keys()}")

    # print("Fund metadata:", fund_metadata)
    fund_summary = generate_fund_summary(fund_metadata)
    if verbose == True:
        print(f"Fund summary: {fund_summary}")
    return {"summary": fund_summary, "metadata": fund_metadata}

