import requests
import json
import xml.etree.ElementTree as ET

def test_pubmed():
    query = "Lyme Disease diagnosis treatment guidelines"
    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query}&retmode=json&retmax=3"
    res = requests.get(search_url).json()
    pmids = res["esearchresult"]["idlist"]
    print("PMIDs:", pmids)
    
    fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={','.join(pmids)}&retmode=xml"
    res2 = requests.get(fetch_url)
    root = ET.fromstring(res2.content)
    
    for article in root.findall(".//PubmedArticle"):
        pmid = article.find(".//PMID").text
        title = article.find(".//ArticleTitle").text
        abstract_elem = article.find(".//AbstractText")
        abstract = abstract_elem.text if abstract_elem is not None else "No abstract"
        
        # Get year
        year_elem = article.find(".//PubDate/Year")
        year = year_elem.text if year_elem is not None else "Unknown year"
        
        print(f"PMID: {pmid}, Year: {year}")
        print(f"Title: {title}")
        print(f"Abstract: {abstract[:100]}...")
        print("-" * 20)

if __name__ == "__main__":
    test_pubmed()
