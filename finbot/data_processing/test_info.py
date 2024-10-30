def check_SEDOL_ISIN_launch_date(dictionary, filename):
    isin = dictionary['ISIN']
    sedol = dictionary['SEDOL']
    launch_date = dictionary['launch_date']

    if not isinstance(isin,str):
        print(f"{filename}: Isin is {type(isin)}")
        dictionary['ISIN'] = 'N/A'
    
    if not isinstance(sedol, str):
        print(f"{filename}: Sedol is {type(sedol)}")
        dictionary['SEDOL'] = 'N/A'

    if not isinstance(launch_date, str):
        print(f"{filename}: Launch is {type(launch_date)}")
        dictionary['launch_date'] = 'N/A'
    
    return dictionary