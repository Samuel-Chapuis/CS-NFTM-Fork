#  python code to retrieve data jornals.
# By Lucia FERNANDEZ
# Git Source inspiration: https://github.com/filiperusso/BDMA/tree/main/UPC2025/BDS 

import springernature_api_client.openaccess as openaccess
import json
import requests
from bs4 import BeautifulSoup
import time
import re


apiKey = 'your_API_key_granted_in_springernature_website'
openaccess_client = openaccess.OpenAccessAPI(api_key = apiKey)

# build your query
query = "((keyword:'NTM' AND keyword:'Neural Turing Machine') OR (keyword:'NFTM' OR keyword:'Neural Field Turing Machine') AND type:{(Journal)} AND onlinedatefrom:2020-01-01 AND onlinedateto:2025-10-01"

result = openaccess_client.search(q=query, p=25, s=1, fetch_all=False, is_premium=False)


# Show results
print(f"Total number of results is {len(result['records'])} \n---")
for record in result['records']:
    print(f"Title: {record.get('title')}")
    print("---")



# Get ISSN or fall back to eISSN if ISSN is empty
for record in result['records']:
    print(f"Title: {record.get('title')}")
    issn = record.get('issn', '') or record.get('eIssn', '')
    record['key_issn'] = issn if issn else 'Not available'
    print(f"key_issn: {record['key_issn']}")
    print("---")


with open('bulkArticles.json', 'w') as f:
    json.dump(result, f)




# Read the bulk file of articles downloaded from the API
with open('bulkArticles.json', 'r') as f:
    data = json.load(f)        




def get_scimago_metrics(issn):
    """Get journal metrics from SCImago by following their search→profile workflow"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    
    try:
        # Step 1: Perform ISSN search
        search_url = f"https://www.scimagojr.com/journalsearch.php?q={issn}"
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Check for CAPTCHA
        if "captcha" in response.text.lower():
            return {'error': 'CAPTCHA triggered - Try manual search first'}
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Step 2: Find the journal link in search results
        results_div = soup.find('div', class_='search_results')
        if not results_div:
            return {'error': 'No search results found'}
            
        journal_link = results_div.find('a', href=True)
        if not journal_link:
            return {'error': 'Journal link not found in results'}
        
        # Step 3: Follow to journal profile page
        journal_url = "https://www.scimagojr.com/" + journal_link['href']
        time.sleep(3)  # Be polite
        
        journal_response = requests.get(journal_url, headers=headers, timeout=10)
        journal_soup = BeautifulSoup(journal_response.text, 'html.parser')
        
        # Step 4: Extract metrics - UPDATED H-INDEX SELECTOR
        h_index_elements = journal_soup.find_all('p', class_='hindexnumber')
        h_index = h_index_elements[1].text.strip() if len(h_index_elements) > 1 else 'N/A'
        sjr = h_index_elements[0].text.strip() if len(h_index_elements) > 1 else 'N/A'
        
        return {
            'journal': journal_link.get_text(strip=True),
            'h_index': h_index,
            'sjr': sjr
        }
        
    except Exception as e:
        return {'error': f'SCImago processing failed: {str(e)}'}

def get_journal_metrics(issn):
    """Main function to get metrics for a journal ISSN"""
    # Clean ISSN format
    issn = re.sub(r'[^0-9X]', '', issn.upper())
    if len(issn) != 8:
        print(f"error: key_issn {issn} Invalid ISSN format (must be 8 characters)")
        return {'error': 'Invalid ISSN format (must be 8 characters)'}
    
    # Try SCImago first
    scimago_result = get_scimago_metrics(issn)
    if not scimago_result.get('error'):
        return {'source': 'SCImago', 'issn': issn, **scimago_result}

    print(f"error: Journal not found in SCImago, key_issn: {issn}")
    return {'error': 'Journal not found in SCImago', 'issn': issn}

# Augmenting the papers with their journals h-index
if __name__ == "__main__":
    
    # some issns for testing, use if needed
    """
    test_issns = [
        "2045-2322",  # Scientific Reports
        "2197-9987",  # Journal of Computers in Education
        "1234-5678"   # Invalid ISSN
    ]
    """
    
    for record in data['records']:
        print(f"Title: {record.get('title')}")

        # uncomment if key_issn was not addressed earlier, for us key_issn is issn that fallsback to eIssn when issn is not available
        #issn = record.get('issn', '') or record.get('eIssn', '')
        #record['key_issn'] = issn if issn else 'Not available'
        
        print(f"\nFetching metrics for ISSN/eISSN: {record['key_issn']}:")
        start_time = time.time()
        metrics = get_journal_metrics(record['key_issn'])
        h_index = metrics.get('h_index', 9999) if metrics else 9999
        print(f"h_index {h_index}")
        record['h_index'] = h_index
        print(f"Results took ({time.time()-start_time:.2f}s)")
        print("---")
        
        #print(metrics)
        time.sleep(5)  # Rate limiting    





with open('bulkArticlesAugmented.json', 'w') as f:
    json.dump(data, f)        