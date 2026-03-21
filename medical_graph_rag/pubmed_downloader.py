"""Pubmeddownloader module."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass

from Bio import Entrez
from tqdm.asyncio import tqdm

logger = logging.getLogger(__name__)


@dataclass
class PubMedEntrezDownloader:
    """PubMedEntrezDownloader class."""

    email: str
    api_key: str | None = None

    def __post_init__(self):
        """Initialize post_init."""
        Entrez.email = self.email
        if self.api_key:
            Entrez.api_key = self.api_key

    async def search_pubmed(
        self,
        query,
        max_results=500,
        date_from=None,
        date_to=None,
        sort_order="relevance",
        publication_types=None,
    ):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._sync_search_pubmed,
            query,
            max_results,
            date_from,
            date_to,
            sort_order,
            publication_types,
        )
        logger.info(f"Found {len(result)} PMIDs")
        return result

    async def fetch_article_details(self, pmids, batch_size=100):
        """Fetch article details."""
        if not pmids:
            logger.warning("No PMIDs to fetch")
            return []

        logger.info(f"Fetching details for {len(pmids)} articles")
        loop = asyncio.get_event_loop()

        tasks = []
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i : i + batch_size]
            task = loop.run_in_executor(None, self._sync_fetch_batch, batch_pmids)
            tasks.append(task)

        batch_results = await tqdm.gather(*tasks, desc="Fetching articles")

        articles = []
        for result in batch_results:
            if isinstance(result, list):
                articles.extend(result)

        logger.info(f"Successfully fetched {len(articles)} articles")
        return articles

    @staticmethod
    def _build_date_filter(date_from: str | None, date_to: str | None) -> str:
        """Build a PubMed date-range filter string."""
        if date_from and date_to:
            return f" AND {date_from}[PDAT]:{date_to}[PDAT]"
        if date_from:
            return f" AND {date_from}[PDAT]:3000[PDAT]"
        if date_to:
            return f" AND 1900[PDAT]:{date_to}[PDAT]"
        return ""

    def _sync_search_pubmed(
        self, query, max_results, date_from, date_to, sort_order, publication_types
    ):
        """Search pubmed synchronously."""
        is_empty_query = not query or query.strip() == ""
        search_term = "research[Title/Abstract]" if is_empty_query else query

        if date_from or date_to:
            search_term += self._build_date_filter(date_from, date_to)

        if publication_types:
            pub_filter = " OR ".join(
                [f'"{pt}"[Publication Type]' for pt in publication_types]
            )
            search_term += f" AND ({pub_filter})"

        if is_empty_query:
            sort_order = "pub_date"

        logger.info(f"Search term: {search_term}")
        handle = Entrez.esearch(
            db="pubmed", term=search_term, retmax=max_results, sort=sort_order
        )
        search_results = Entrez.read(handle)
        handle.close()

        return search_results["IdList"]

    def _sync_fetch_batch(self, batch_pmids: list[str]) -> list[dict]:
        """Fetch a batch of articles by PubMed ID and parse them."""
        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(batch_pmids),
                rettype="xml",
                retmode="xml",
            )
            records = Entrez.read(handle)
            handle.close()

            articles = []
            for record in records["PubmedArticle"]:
                article_data = self._parse_article(record)
                if article_data:
                    articles.append(article_data)

            time.sleep(0.34)
            return articles

        except Exception as e:
            logger.error(
                f"Batch fetch failed for {len(batch_pmids)} PMIDs: {e}",
                exc_info=True,
            )
            return []

    @staticmethod
    def _extract_abstract(article: dict) -> str:
        """Extract concatenated abstract text from an article dict."""
        if "Abstract" not in article or "AbstractText" not in article["Abstract"]:
            return ""
        return " ".join(str(t) for t in article["Abstract"]["AbstractText"])

    @staticmethod
    def _extract_identifiers(article: dict) -> tuple[str, str]:
        """Extract DOI and PMC ID from the ELocationID list."""
        doi = pmc_id = ""
        for eloc in article.get("ELocationID", []):
            eid_type = eloc.attributes.get("EIdType", "")
            if eid_type == "doi":
                doi = str(eloc)
            elif eid_type == "pmc":
                pmc_id = str(eloc)
        return doi, pmc_id

    @staticmethod
    def _extract_pub_date_info(journal_issue: dict) -> tuple[str, str, str]:
        """Extract year, month, day from a JournalIssue dict."""
        pub_date_info = journal_issue.get("PubDate", {})
        return (
            pub_date_info.get("Year", ""),
            pub_date_info.get("Month", ""),
            pub_date_info.get("Day", ""),
        )

    @staticmethod
    def _extract_authors(article: dict) -> str:
        """Extract formatted author string from an article dict."""
        authors_list = article.get("AuthorList", [])
        return "; ".join(
            f"{a.get('LastName', '')} {a.get('ForeName', '')}".strip()
            for a in authors_list
            if "LastName" in a
        )

    def _parse_article(self, record: dict) -> dict | None:
        """Parse a PubmedArticle record into structured data."""
        try:
            citation = record["MedlineCitation"]
            article = citation["Article"]

            pmid = str(citation.get("PMID", ""))
            title = str(article.get("ArticleTitle", "")).strip()
            abstract = self._extract_abstract(article)
            authors = self._extract_authors(article)

            journal_info = article.get("Journal", {})
            journal_title = str(journal_info.get("Title", ""))
            journal_issue = journal_info.get("JournalIssue", {})
            volume = str(journal_issue.get("Volume", ""))
            issue = str(journal_issue.get("Issue", ""))
            year, month, day = self._extract_pub_date_info(journal_issue)

            doi, pmc_id = self._extract_identifiers(article)

            mesh_terms = [
                str(m["DescriptorName"]) for m in citation.get("MeshHeadingList", [])
            ]
            pub_types = [str(pt) for pt in article.get("PublicationTypeList", [])]

            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "journal": journal_title,
                "volume": volume,
                "issue": issue,
                "year": year,
                "month": month,
                "day": day,
                "pub_date": year,
                "doi": doi,
                "pmc_id": pmc_id,
                "mesh_terms": "; ".join(mesh_terms),
                "publication_types": "; ".join(pub_types),
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "doi_url": f"https://doi.org/{doi}" if doi else "",
            }

        except (KeyError, TypeError) as e:
            logger.warning(f"Could not parse article record: {e}")
            return None

    def save_to_json(self, articles, filename):
        """Save To Json method."""
        if articles:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(articles, f, indent=2, ensure_ascii=False)
