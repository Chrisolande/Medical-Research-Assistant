from Bio import Entrez
import pandas as pd
import time
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional
from tqdm.asyncio import tqdm
@dataclass
class PubMedEntrezDownloader:
    email: str
    api_key: Optional[str] = None

    def __post_init__(self):
        Entrez.email = self.email
        if self.api_key:
            Entrez.api_key = self.api_key   

    async def search_pubmed(self, query, max_results=500, date_from=None, date_to=None, sort_order="relevance", publication_types=None):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._sync_search_pubmed, query, max_results, date_from, date_to, sort_order, publication_types)
        print(f"Found {len(result)} PMIDs")
        return result

    async def fetch_article_details(self, pmids, batch_size=100):
        if not pmids:
            print("No PMIDs to fetch")
            return []
            
        print(f"Fetching details for {len(pmids)} articles")
        loop = asyncio.get_event_loop()
        
        tasks = []
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            task = loop.run_in_executor(None, self._sync_fetch_batch, batch_pmids)
            tasks.append(task)
        
        batch_results = await tqdm.gather(*tasks, desc="Fetching articles")
        
        articles = []
        for result in batch_results:
            if isinstance(result, list):
                articles.extend(result)
        
        print(f"Successfully fetched {len(articles)} articles")
        return articles

    def _sync_search_pubmed(self, query, max_results, date_from, date_to, sort_order, publication_types):
        # Handle case where no specific query is provided - use a broad search instead of "*"
        if not query or query.strip() == "":
            search_term = "research[Title/Abstract]"  # Broad but valid search
        else:
            search_term = query

        if date_from or date_to:
            if date_from and date_to:
                search_term += f' AND {date_from}[PDAT]:{date_to}[PDAT]'
            elif date_from:
                search_term += f' AND {date_from}[PDAT]:3000[PDAT]'
            elif date_to:
                search_term += f' AND 1900[PDAT]:{date_to}[PDAT]'

        if publication_types:
            pub_filter = ' OR '.join([f'"{pt}"[Publication Type]' for pt in publication_types])
            search_term += f' AND ({pub_filter})'

        # For diverse papers, sort by date to get recent papers first
        if not query or query.strip() == "":
            sort_order = "pub_date"

        print(f"Search term: {search_term}")
        handle = Entrez.esearch(db="pmc", term=search_term, retmax=max_results, sort=sort_order)
        search_results = Entrez.read(handle)
        handle.close()

        return search_results["IdList"]

    def _sync_fetch_batch(self, batch_pmids):
        try:
            handle = Entrez.esummary(db="pmc", id=",".join(batch_pmids))
            summaries = Entrez.read(handle)
            handle.close()

            handle = Entrez.efetch(db="pmc", id=",".join(batch_pmids), rettype="medline", retmode="xml")
            records = Entrez.read(handle)
            handle.close()

            articles = []
            for summary, record in zip(summaries, records['PubmedArticle']):
                article_data = self._parse_article(summary, record)
                if article_data:
                    articles.append(article_data)
            
            time.sleep(0.34)
            return articles

        except Exception:
            return []

    def _parse_article(self, summary, record):
        """Parse article summary and record into structured data"""
        pmid = str(summary.get('Id', ''))
        title = summary.get('Title', '').strip()
        journal = summary.get('Source', '')
        pub_date = summary.get('PubDate', '')
        
        authors_list = summary.get('AuthorList', [])
        authors = '; '.join([author for author in authors_list]) if authors_list else ''
        
        article = record['MedlineCitation']['Article']
        
        # Abstract
        abstract = ''
        if 'Abstract' in article and 'AbstractText' in article['Abstract']:
            abstract_texts = [str(abs_text) for abs_text in article['Abstract']['AbstractText']]
            abstract = ' '.join(abstract_texts)
        
        # Publication details
        journal_info = article.get('Journal', {})
        journal_title = journal_info.get('Title', journal)
        
        journal_issue = journal_info.get('JournalIssue', {})
        volume = journal_issue.get('Volume', '')
        issue = journal_issue.get('Issue', '')
        
        pub_date_info = journal_issue.get('PubDate', {})
        year = pub_date_info.get('Year', '')
        month = pub_date_info.get('Month', '')
        day = pub_date_info.get('Day', '')
        
        # Identifiers
        doi = pmc_id = ''
        if 'ELocationID' in article:
            for eloc in article['ELocationID']:
                if eloc.attributes.get('EIdType') == 'doi':
                    doi = str(eloc)
                elif eloc.attributes.get('EIdType') == 'pmc':
                    pmc_id = str(eloc)
        
        # MeSH terms
        mesh_terms = []
        if 'MeshHeadingList' in record['MedlineCitation']:
            mesh_terms = [str(mesh['DescriptorName']) for mesh in record['MedlineCitation']['MeshHeadingList']]
        
        # Publication types
        pub_types = [str(pt) for pt in article.get('PublicationTypeList', [])]
        
        return {
            'pmid': pmid, 'title': title, 'abstract': abstract, 'authors': authors,
            'journal': journal_title, 'volume': volume, 'issue': issue,
            'year': year, 'month': month, 'day': day, 'pub_date': pub_date,
            'doi': doi, 'pmc_id': pmc_id,
            'mesh_terms': '; '.join(mesh_terms),
            'publication_types': '; '.join(pub_types),
            'pubmed_url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            'doi_url': f"https://doi.org/{doi}" if doi else ''
        }

    def save_to_csv(self, articles, filename):
        if articles:
            pd.DataFrame(articles).to_csv(filename, index=False, encoding='utf-8')

    def save_to_excel(self, articles, filename):
        if articles:
            pd.DataFrame(articles).to_excel(filename, index=False, engine='openpyxl')

    def save_to_json(self, articles, filename):
        if articles:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(articles, f, indent=2, ensure_ascii=False)