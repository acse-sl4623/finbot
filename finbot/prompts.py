from langchain.prompts import PromptTemplate

# Task Classification Prompt Template
task_classification_template = PromptTemplate.from_template(
    """Given the user question below, classify it as a task to
    `describe`, `compare`, `analyze` or is it a `specific information request`.
    
    Do not respond with more than one word.
    
    <question>
    {question}
    </question>
    Classification:""",
)

# Infer k from query Prompt Template
infer_k_from_query_template = PromptTemplate.from_template(
    """Given the user question below, identify the number of funds the user expects.
    
    Respond only with a single number, like "1", "2", "3", etc.
    
    <question>
    {question}
    </question>
    Number of funds:""",
)

# Subject Classification Prompt Template
subject_classification_template = PromptTemplate.from_template(
    """Given the user question below, classify it as a question regarding
    `performance`, `sector allocation`, `region allocation`, `asset allocation`. 

    Exposure is a synonym for allocation.
    
    Do not respond with more than one word.
    
    <question>
    {question}
    </question>
    Classification:""",
)

# Extract Fund Names Prompt Template
extract_fund_names_template = PromptTemplate.from_template(
    """Extract the fund names from the user question below.
    
    Only respond with the fund names seperated by a comma.

    <question>
    {question}
    </question>
    
    Fund names:"""
)

# General Template
general_template = PromptTemplate.from_template("""
Extract the relevant information from the investment fund factsheets provided to answer the following question:
                               
Query: {question}
                                                
Investment Fund Factsheet: {context}

Always specify the fund name or names for which the answer corresponds to.

""")

# List Template
list_template = PromptTemplate.from_template("""
Query: {question}
                                                
Funds in knowledge base: {context}
""")

# Guided Performance Description Template
guided_description_perf_template = PromptTemplate.from_template("""                                             
Answer the following question in the provided style.
                                                                
Question: {question}   
                                                    
Style: Take on the role of a financial analyst whose task is to present a fund's
historical performance and peer performance and provide insights to a client.
Provide a short concise summary in a narrative format of the information relevant to the question in the context.

In your summary, consider the following:

0. Always start the description by naming the fund
1. Identify notable changes in the historical performance of the fund over the various time periods provided.
2. Identify if the fund has generally outperformed or underperformed its benchmark (if the benchmark exists), notably in the last 1 year.
3. Identify notable changes in the rank within sector and quartile rank over the time periods provided, which are indicative of the
the fund's performance compared to its peers.
4. Compute for the 1 year return: (use only 1-2 sentences)
    a) active performance of the fund compared to its benchmark over the time periods provided
    b) percentile rank of the fund compared to its peers over the time periods provided

When computing state the formulas first and then perform the computation. 
Incorporate the results of the computations into your analysis, making it clear how the provided factsheets inform your insights.
Always start the description by naming the fund and the description should not have more than 200 words.
                                                                
Investment Fund Factsheets: {context}
                                                    
""")

# Guided Exposure Description Template
guided_description_exp_template = PromptTemplate.from_template("""
Analyze the provided {section} allocation for {fund_name} in the context of it's investment objective.
                                                               
Question: Is the following exposure in line with the investment objective?

{section} allocation table: {allocation_table}               
Investment Objective: {inv_obj}

When answering the question make it clear how the provided allocation table and investment objective inform your decision.
Always start the description by naming the fund and the description should not have more than 200 words.                

""")

# Simple Performance Comparison Template (2 funds only)
compare_funds_perf_template = PromptTemplate.from_template("""
Use the performance information in the context to provide a comprehensive consise comparative narrative between the two investment funds.
                                                      
Include in the comparative narrative the following: 
- Compare the performance of the funds over the time periods provided.
- Specify which fund has outperformed it's benchmark more consistently.
- Specify which fund has performed better relative to its peers given the quartile and rank within sector.
                                            
Context: {context}

Make it clear how the contexts informs your insights.
The narrative should not have more than 200 words.

""")

# Simple Exposure Comparison Template (2 funds only)
compare_funds_exp_template = PromptTemplate.from_template("""
Use the information in the context to provide a comprehensive consise comparative narrative between the two investment funds.
                                                      
Include in the comparative narrative the following: 
- Difference in allocation between the two funds.
                                                      
Context: {context}

Make it clear how the contexts informs your insights.
The narrative should not have more than 200 words.

""")

# Peer Performance Analysis Template
analyze_peer_perf_score_template = PromptTemplate.from_template("""
The fund ({fund_name}) has the following performance metrics relative to the peer group: {peers} with average 1 year return of {avg_1yr_return}.

Peers in peer group: {peers}

Fund's 1 year return percentile rank: {score_universe_rank}
                                                                
Interpret the fund's performance relative to the peer group.
                                                                
Include the names of the funds in the peer group in the analysis.
The narrative should not have more than 200 words.

""")

# Peer Exposure Analysis Template
analyze_peer_exp_score_template = PromptTemplate.from_template("""
The fund ({fund_name}) has top {section} exposure to {top_exposure_name} with {top_exposure_percentage}% invested.
                                                               
Relative to the following peer group with similar investment objective we observe an average exposure to {top_exposure_name} of {average_exposure}

Peers in peer group: {peers}
                                                               
Relative to this peer group, {fund_name}'s exposure to {top_exposure_name} has a percentile rank of {percentile_rank}.
                                                                
Interpret the fund's exposure relative to the peer group.
The narrative should not have more than 200 words.

""")

# (Legacy) Simple Description Template (used in route only)
simple_description_template = PromptTemplate.from_template("""
Answer the following question in the provided style.

Question: {question}                                                 

Style: Extract the relevant information from the investment fund factsheets to answer the question.                        
Summarize the relevant information and start the answer with "Description: "
                                                           
Investment Fund Factsheet: {context}
Answer:
""")

# (Legacy) Simple Comparison Template (used in route only)
simple_comparison_template = PromptTemplate.from_template("""
Compare the allocation and performance between the two investment fund factsheets provided in the context.
                                                                                                                                                                   
Context: {context}

""")