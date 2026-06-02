import os
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import pathlib

_HERE = pathlib.Path(__file__).parent.parent.parent
CACHE_FILE = _HERE / "data" / "pubmed_cache.json"

class PubMedRetriever:
    def __init__(self, cache_file=CACHE_FILE):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.base_url_search = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.base_url_fetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[PubMed] Error loading cache: {e}")
        return {}

    def _save_cache(self):
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"[PubMed] Error saving cache: {e}")

    def _search_pmids(self, query: str, max_results: int) -> list[str]:
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "json",
            "sort": "relevance"
        }
        query_string = urllib.parse.urlencode(params)
        url = f"{self.base_url_search}?{query_string}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'SymptomAssist/1.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
                return data.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            print(f"[PubMed] Error searching PMIDs: {e}")
            return []

    def _fetch_abstracts(self, pmids: list[str]) -> list[dict]:
        if not pmids:
            return []
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml"
        }
        query_string = urllib.parse.urlencode(params)
        url = f"{self.base_url_fetch}?{query_string}"
        
        results = []
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'SymptomAssist/1.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                xml_data = response.read()
                root = ET.fromstring(xml_data)
                
                for article in root.findall(".//PubmedArticle"):
                    pmid = article.findtext(".//PMID")
                    title = article.findtext(".//ArticleTitle") or "No Title Available"
                    
                    # Abstract can be multiple parts
                    abstract_texts = article.findall(".//AbstractText")
                    abstract = " ".join([elem.text for elem in abstract_texts if elem.text])
                    if not abstract:
                        continue # Skip articles without an abstract
                        
                    year = article.findtext(".//PubDate/Year")
                    if not year:
                        year = article.findtext(".//ArticleDate/Year") or "Unknown Year"
                        
                    results.append({
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "year": year
                    })
        except Exception as e:
            print(f"[PubMed] Error fetching abstracts: {e}")
            
        return results

    def retrieve(self, condition: str, max_results: int = 3) -> list[dict]:
        """
        Retrieves abstracts for a given condition. Uses cache if available.
        Returns a list of dicts: {pmid, title, abstract, year}
        """
        query = f'"{condition}"[Title/Abstract] AND (diagnosis OR treatment OR guidelines)'
        
        cache_key = condition.lower().strip()
        if cache_key in self.cache:
            print(f"[PubMed] Cache hit for '{condition}'")
            return self.cache[cache_key][:max_results]
            
        print(f"[PubMed] Fetching live data for '{condition}'...")
        pmids = self._search_pmids(query, max_results=max_results + 2) # Fetch a couple extra in case some lack abstracts
        if not pmids:
            return []
            
        abstracts = self._fetch_abstracts(pmids)
        
        # Take the top ones
        results = abstracts[:max_results]
        
        # Save to cache
        if results:
            self.cache[cache_key] = results
            self._save_cache()
            
        return results
