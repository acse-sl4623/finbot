import re
# 2a. Testing the extraction pipeline
# Test preprocessing
def check_perf_dict_keys (dictionary, expected_keys, filename):
    all_keys = list(dictionary.keys())

    if len(all_keys) != len(expected_keys):
        print(f"Number of keys do not match for {filename}")
        print(f"Expected keys: {expected_keys}")
        print(f"Actual keys: {all_keys}")
        return False
    for i in range(len(all_keys)):
        if all_keys[i] != expected_keys[i]:
            print(f"Key {i}: {all_keys[i]} does not match {expected_keys[i]} for {filename}")
            return False
    
    print(f"PASSED: {filename} All keys are valid")
    return True

def check_perf_dict_values(data, expected_keys, filename):
    # Define regex patterns for validation
    performance_pattern =  r'^[+-]\d+\.\d+$'  # +digit.digit
    rank_pattern = r'^\d+ / \d+$'             # digit digit / digit digit
    quartile_pattern = r'^\d$'                # single digit
    for key, values in data.items():
        # Check the specific conditions for 5 years and 3 years
        if key == expected_keys[-2]:
            fund_3yrs = values['Fund']
            key_5yrs = expected_keys[-1]
            fund_5yrs = data[key_5yrs]['Fund']

            if 'Benchmark' in values.keys():
                benchmark_3yrs = values['Benchmark']
                benchmark_5yrs = data[key_5yrs]['Benchmark']
                if (fund_3yrs == 'n/a' and benchmark_3yrs != 'n/a') and (fund_5yrs == 'n/a' and benchmark_5yrs != 'n/a'):
                    print(f"Likely young fund: {filename}")
                    return True
            else:
                if fund_3yrs == 'n/a' and fund_5yrs == 'n/a':
                    print(f"Likely young fund: {filename}")
                    return True

        elif key == expected_keys[-1]: # 5yrs or 48m-60m
            fund_5yrs = values['Fund']
            if 'Benchmark' in values.keys():
                benchmark_5yrs = values['Benchmark']
                if fund_5yrs == 'n/a' and benchmark_5yrs != 'n/a':
                    print(f"Likely young fund: {filename}")
                    return True
            else:
                if fund_5yrs == 'n/a':
                    print(f"Likely young fund: {filename}")
                    return True

        else:
            for sub_key, value in values.items():
                # Ensure value is a string before validating
                if not isinstance(value, str):
                    print(f"{filename}: Value for '{sub_key}' in '{key}' is not a string: {value}")
                    return False

                if sub_key == 'Fund':
                    # Check if the value matches pattern or is n/a
                    if not re.match(performance_pattern, value) and value != 'n/a':
                        print(f"{filename}:Invalid Fund value for '{key}': '{value}'")
                        return False
                elif sub_key == 'Benchmark':
                    # Check if the value matches pattern or is n/a
                    if not re.match(performance_pattern, value) and value != 'n/a':
                        print(f"{filename}:Invalid Benchmark value for '{key}': '{value}'")
                        return False
                elif sub_key == 'Rank within sector':
                    # Check if the value matches pattern or is n/a
                    if not re.match(rank_pattern, value) and value != 'n/a':
                        print(f"{filename}:Invalid Rank within sector value for '{key}': '{value}'")
                        return False
                elif sub_key == 'Quartile':
                    # Check if the value matches pattern or is n/a
                    if not re.match(quartile_pattern, value) and value != 'n/a':
                        print(f"{filename}:Invalid Quartile value for '{key}': '{value}'")
                        return False
    print(f"PASSED: {filename} All values are valid")
    return True