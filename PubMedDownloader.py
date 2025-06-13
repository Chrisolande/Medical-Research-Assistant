from Bio import Entrez
import pandas as pd
import time
import json
from datetime import datetime
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

@dataclass
class PubMedEntrezDownloader:
    email: str
    api_key: Optional[str] = None

    def __post_init__(self):
        Entrez.email = self.email
        if self.api_key:
            Entrez.api_key = self.api_key   

    def search_pubmed(self, query, max_results=500, date_from=None, date_to=None, sort_order="relevance", publication_types=None):
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

        handle = Entrez.esearch(db="pmc", term=search_term, retmax=max_results, sort=sort_order)
        search_results = Entrez.read(handle)
        handle.close()

        return search_results["IdList"]

    def fetch_article_details(self, pmids, batch_size=100):
        articles = []

        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]

            try:
                handle = Entrez.esummary(db="pmc", id=",".join(batch_pmids))
                summaries = Entrez.read(handle)
                handle.close()

                handle = Entrez.efetch(db="pmc", id=",".join(batch_pmids), rettype="medline", retmode="xml")
                records = Entrez.read(handle)
                handle.close()

                for summary, record in zip(summaries, records['PubmedArticle']):
                    article_data = self._parse_article(summary, record)
                    if article_data:
                        articles.append(article_data)
                
                time.sleep(0.34)

            except Exception:
                continue
        
        return articles

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

    def advanced_search(self, **kwargs):
        """
        Perform advanced search with multiple parameters
        
        Available parameters:
        - query: Main search terms
        - author: Author name
        - journal: Journal name
        - mesh_terms: MeSH terms list
        - title_words: Words that must appear in title
        - abstract_words: Words that must appear in abstract
        - date_from/date_to: Date range
        - publication_types: List of publication types
        - languages: List of languages
        - max_results: Maximum results
        """

        search_parts = []
        # Main query
        if "query" in kwargs:
            search_parts.append(kwargs["query"])

        # Author
        if "author" in kwargs:
            search_parts.append(f'"{kwargs["author"]}"[Author]')
        
        # Journal
        if "journal" in kwargs:
            search_parts.append(f'"{kwargs["journal"]}"[Journal]')

        # Mesh terms
        if "mesh_terms" in kwargs:
            mesh_queries = [f'"{term}"[MeSH Terms]' for term in kwargs['mesh_terms']]
            search_parts.append(f'({" OR ".join(mesh_queries)})')

        # Title words
        if 'title_words' in kwargs:
            title_queries = [f'"{word}"[Title]' for word in kwargs['title_words']]
            search_parts.append(f'({" AND ".join(title_queries)})')
        
        # Abstract words
        if 'abstract_words' in kwargs:
            abstract_queries = [f'"{word}"[Abstract]' for word in kwargs['abstract_words']]
            search_parts.append(f'({" AND ".join(abstract_queries)})')
        
        # Languages
        if 'languages' in kwargs:
            lang_queries = [f'"{lang}"[Language]' for lang in kwargs['languages']]
            search_parts.append(f'({" OR ".join(lang_queries)})')
        
        # Combine all parts
        full_query = ' AND '.join(search_parts)

        return self.search_pubmed(
                query=full_query,
                max_results=kwargs.get('max_results', 500),
                date_from=kwargs.get('date_from'),
                date_to=kwargs.get('date_to'),
                publication_types=kwargs.get('publication_types')
            )

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