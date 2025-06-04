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
        self.email = self.email

        if self.api_key:
            Entrez.api_key = self.api_key

    def search_pubmed(self, query, max_results = 500, date_from = None, date_to = None, sort_order = "relevance", publication_types = None):
        
        search_term = query

        # Add date filters
        if date_from or date_to:
            if date_from and date_to:
                search_term += f' AND {date_from}[PDAT]:{date_to}[PDAT]'
            elif date_from:
                search_term += f' AND {date_from}[PDAT]:3000[PDAT]'
            elif date_to:
                search_term += f' AND 1900[PDAT]:{date_to}[PDAT]'

        # Add publication type filters
        if publication_types:
            pub_filter = ' OR '.join([f'"{pt}"[Publication Type]' for pt in publication_types])
            search_term += f' AND ({pub_filter})'
        
        print(f"Searching PubMed with query: {search_term}")

        try: 
            # Perform the search
            handle = Entrez.esearch(
                db = "pmc",
                term=search_term,
                retmax=max_results,
                sort=sort_order
            )

            search_results = Entrez.read(handle)

            handle.close()

            pmids = search_results["IdList"]
            count = int(search_results["Count"])
            
            print(f"Found {count} total articles, retrieving {len(pmids)} IDs")
            return pmids

        except Exception as e:
            print(f"Search error: {e}")
            return []


    def fetch_article_details(self, pmids, batch_size = 100):
        articles = []

        # Process in batches
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]

            print(f"Fetching batch {i//batch_size + 1}/{(len(pmids)-1)//batch_size + 1} "
                    f"({len(batch_pmids)} articles)...")

            try:
                # Fetch article summaries first
                handle = Entrez.esummary(db = "pmc", id = ",".join(batch_pmids))
                summaries = Entrez.read(handle)
                handle.close()

                # Fetch full abstract records
                handle = Entrez.efetch(
                    db = "pmc",
                    id = ",".join(batch_pmids),
                    rettype = "medline",
                    retmode = "xml"
                )

                records = Entrez.read(handle)
                handle.close()

                # Parse articles
                for summary, record in zip(summaries, records['PubmedArticle']):
                    article_data = self._parse_article(summary, record)
                    if article_data:
                        articles.append(article_data)
                
                # Simple rate limiter
                #TODO: Implement the exponential backoff if need be
                time.sleep(0.34) # 3 requests per second

            except Exception as e:
                print(f"Error fetching batch {i//batch_size + 1}: {e}")
                continue
        
        return articles

    def _parse_article(self, summary, record):
        """Parse article summary and record into structured data"""
        try:
            # Basic info from summary
            pmid = str(summary.get('Id', ''))
            title = summary.get('Title', '').strip()
            
            # Journal info
            journal = summary.get('Source', '')
            pub_date = summary.get('PubDate', '')
            
            # Authors from summary
            authors_list = summary.get('AuthorList', [])
            authors = '; '.join([author for author in authors_list]) if authors_list else ''
            
            # Extract more details from full record
            article = record['MedlineCitation']['Article']
            
            # Abstract
            abstract = ''
            if 'Abstract' in article:
                abstract_texts = []
                if 'AbstractText' in article['Abstract']:
                    for abs_text in article['Abstract']['AbstractText']:
                        if isinstance(abs_text, str):
                            abstract_texts.append(abs_text)
                        else:
                            # Handle structured abstracts with labels
                            abstract_texts.append(str(abs_text))
                abstract = ' '.join(abstract_texts)
            
            # Publication details
            journal_info = article.get('Journal', {})
            journal_title = journal_info.get('Title', journal)
            
            # Volume and issue
            journal_issue = journal_info.get('JournalIssue', {})
            volume = journal_issue.get('Volume', '')
            issue = journal_issue.get('Issue', '')
            
            # Publication date details
            pub_date_info = journal_issue.get('PubDate', {})
            year = pub_date_info.get('Year', '')
            month = pub_date_info.get('Month', '')
            day = pub_date_info.get('Day', '')
            
            # DOI and other identifiers
            doi = ''
            pmc_id = ''
            
            if 'ELocationID' in article:
                for eloc in article['ELocationID']:
                    if eloc.attributes.get('EIdType') == 'doi':
                        doi = str(eloc)
                    elif eloc.attributes.get('EIdType') == 'pmc':
                        pmc_id = str(eloc)
            
            # Keywords/MeSH terms
            mesh_terms = []
            if 'MeshHeadingList' in record['MedlineCitation']:
                for mesh in record['MedlineCitation']['MeshHeadingList']:
                    descriptor = mesh['DescriptorName']
                    mesh_terms.append(str(descriptor))
            
            # Publication types
            pub_types = []
            if 'PublicationTypeList' in article:
                pub_types = [str(pt) for pt in article['PublicationTypeList']]
            
            return {
                'pmid': pmid,
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'journal': journal_title,
                'volume': volume,
                'issue': issue,
                'year': year,
                'month': month,
                'day': day,
                'pub_date': pub_date,
                'doi': doi,
                'pmc_id': pmc_id,
                'mesh_terms': '; '.join(mesh_terms),
                'publication_types': '; '.join(pub_types),
                'pubmed_url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                'doi_url': f"https://doi.org/{doi}" if doi else ''
            }
            
        except Exception as e:
            print(f"Error parsing article {pmid}: {e}")
            return None

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

    def get_article_statistics(self, articles):
        """Generate basic statistics about downloaded articles"""
        if not articles:
            return {}
        
        df = pd.DataFrame(articles)
        
        stats = {
            'total_articles': len(articles),
            'articles_with_abstracts': len(df[df['abstract'].str.len() > 0]),
            'date_range': {
                'earliest': df['year'].min(),
                'latest': df['year'].max()
            },
            'top_journals': df['journal'].value_counts().head(10).to_dict(),
            'publication_types': df['publication_types'].value_counts().head(10).to_dict(),
            'articles_per_year': df['year'].value_counts().sort_index().to_dict()
        }
        
        return stats

    def save_to_csv(self, articles, filename):
        """Save articles to CSV using pandas"""
        if not articles:
            print("No articles to save")
            return
            
        df = pd.DataFrame(articles)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Saved {len(articles)} articles to {filename}")

    def save_to_excel(self, articles, filename):
        """Save articles to Excel file"""
        if not articles:
            print("No articles to save")
            return
        
        df = pd.DataFrame(articles)
        df.to_excel(filename, index=False, engine='openpyxl')
        print(f"Saved {len(articles)} articles to {filename}")

    def save_to_json(self, articles, filename):
        """Save articles to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(articles)} articles to {filename}")