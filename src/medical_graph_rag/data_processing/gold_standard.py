import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

import backoff
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI

from medical_graph_rag.core.config import LLM_MODEL_NAME, OPENROUTER_API_BASE

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    id: str
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class QATriple:
    question: str
    answer: str
    source_doc_id: str
    context: str = ""
    difficulty: str = "medium"
    question_type: str = "factual"


@dataclass
class LLMConfig:
    model: str = LLM_MODEL_NAME
    api_key: str | None = None
    base_url: str = OPENROUTER_API_BASE
    temperature: float = 0.7
    max_tokens: int = 2000
    max_concurrent: int = 5
    timeout: int = 30


@dataclass
class GoldStandardBuilder:
    documents: list[Document]
    qa_triples: list[QATriple] = field(default_factory=list)
    llm_config: LLMConfig = field(default_factory=LLMConfig)

    def __post_init__(self):
        self.llm = ChatOpenAI(
            model=self.llm_config.model,
            api_key=self.llm_config.api_key,
            openai_api_base=self.llm_config.base_url,
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens,
            request_timeout=self.llm_config.timeout,
        )

    async def generate_questions_with_llm(
        self, num_questions_per_doc: int = 3
    ) -> list[QATriple]:
        logger.info(f"Starting generation for {len(self.documents)} documents")
        success_count = 0
        error_count = 0

        async def process_document(doc: Document):
            nonlocal success_count, error_count
            try:
                result = await self._generate_for_doc_with_retry(
                    doc, num_questions_per_doc
                )
                success_count += 1
                return result
            except Exception as e:
                error_count += 1
                logger.error(f"Failed processing document {doc.id}: {str(e)}")
                return []

        semaphore = asyncio.Semaphore(self.llm_config.max_concurrent)
        tasks = [
            self._throttled_process(doc, semaphore, process_document)
            for doc in self.documents
        ]
        results = await asyncio.gather(*tasks)
        self.qa_triples = [qa for sublist in results for qa in sublist]

        logger.info(
            f"Generation completed. Success: {success_count}, Failures: {error_count}, Total QAs: {len(self.qa_triples)}"
        )
        return self.qa_triples

    async def _throttled_process(self, doc, semaphore, process_fn):
        async with semaphore:
            return await process_fn(doc)

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def _generate_for_doc_with_retry(
        self, doc: Document, num_questions: int
    ) -> list[QATriple]:
        prompt = self._build_qa_prompt(doc.content, num_questions)
        response = await self.llm.agenerate([[HumanMessage(content=prompt)]])
        return self._parse_llm_response(response.generations[0][0].text, doc.id)

    def _build_qa_prompt(self, content: str, num_questions: int) -> str:
        return f"""Generate {num_questions} question-answer pairs from this content:

{content[:3000]}

Return JSON array:
[{{"question": "...", "answer": "...", "context": "...", "difficulty": "easy/medium/hard", "question_type": "factual/inferential/analytical"}}]"""

    def _parse_llm_response(self, raw_response: str, doc_id: str) -> list[QATriple]:
        json_match = re.search(r"(\[.*?\])", raw_response, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON array found")

        qa_data = json.loads(json_match.group(1))
        return [
            QATriple(
                question=item["question"],
                answer=item["answer"],
                source_doc_id=doc_id,
                context=item.get("context", ""),
                difficulty=item.get("difficulty", "medium"),
                question_type=item.get("question_type", "factual"),
            )
            for item in qa_data
            if "question" in item and "answer" in item
        ]

    def export_dataset(self, filepath: str, indent: int = 2) -> None:
        export_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "documents_processed": len(self.documents),
                "qa_pairs_generated": len(self.qa_triples),
                "llm_config": {
                    "model": self.llm_config.model,
                    "temperature": self.llm_config.temperature,
                },
            },
            "qa_pairs": [
                {
                    "id": f"qa_{idx:04d}",
                    "question": qa.question,
                    "answer": qa.answer,
                    "source": qa.source_doc_id,
                    "context": qa.context,
                    "difficulty": qa.difficulty,
                    "type": qa.question_type,
                }
                for idx, qa in enumerate(self.qa_triples)
            ],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=indent, ensure_ascii=False)

        logger.info(f"Dataset exported to {filepath}")

    def get_stats(self) -> dict:
        difficulties = {}
        types = {}
        for qa in self.qa_triples:
            difficulties[qa.difficulty] = difficulties.get(qa.difficulty, 0) + 1
            types[qa.question_type] = types.get(qa.question_type, 0) + 1

        return {
            "total_qa_pairs": len(self.qa_triples),
            "difficulty_distribution": difficulties,
            "question_type_distribution": types,
            "questions_per_document": (
                len(self.qa_triples) / len(self.documents) if self.documents else 0
            ),
        }
