# from ddgs import DDGS
# from langchain.schema import Document
# from langchain_ollama import OllamaLLM
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma


# class Assistant:
#
#     def __init__(self, gen_model="phi3:mini", vect_model="sentence-transformers/all-MiniLM-L6-v2"):
#         self.vect_model = SentenceTransformerEmbeddings(model_name=vect_model)
#         self.llm = OllamaLLM(
#             model="phi3:mini",
#             temperature=0,
#             base_url="http://localhost:11434",
#         )
#         self.prompt = PromptTemplate(
#             input_variables=["context", "question"],
#             template=
#             """ hi, You are an assistant that answers questions about recent news using the provided context from search result.
#             If the answer is not in the context, say "I don't know".
#
#             Context:
#             {context}
#
#             Question:
#             {question}
#
#             Answer (be concise, but a bit verbose):"""
#         )
#
#     @staticmethod
#     def get_search_results(query, num_res=20):
#         with DDGS() as ddgs:
#             raw_results = list(ddgs.text(query, max_results=num_res))
#
#         docs = []
#         for r in raw_results:
#             content = r.get("body", "")
#             if not content.strip():
#                 continue
#             metadata = {"title": r.get("title"), "url": r.get("href")}
#             docs.append(Document(page_content=content, metadata=metadata))
#         return docs
#
#     def get_best_hits(self, docs, query, chunk_size=100, chunk_overlap_ratio=0.2, k=2):
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=int(chunk_size * chunk_overlap_ratio)
#         )
#         chunked_docs = splitter.split_documents(docs)
#         vectordb = Chroma.from_documents(chunked_docs, self.vect_model)
#         retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": k, "lambda_mult": 0.7})
#         retrieved_docs = retriever.get_relevant_documents(query)
#         return retrieved_docs
#
#     def run(self, query, chunk_size=100, top_n_docs=3):
#         docs = self.get_search_results(query)
#         best_hits = self.get_best_hits(docs, query, chunk_size=chunk_size, k=top_n_docs)
#         context_text = "\n\n".join(
#             [f"Title: {d.metadata['title']}\nURL: {d.metadata['url']}\n{d.page_content}" for d in best_hits]
#         )
#         final_prompt = self.prompt.format(context=context_text, question=query)
#
#         print(final_prompt)
#         answer = self.llm.invoke(final_prompt)
#         return answer






# app.py  (replace the Assistant class in your file with this)
from ddgs import DDGS
from langchain.schema import Document
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from collections import defaultdict, Counter
import math
import re
from logger_config import logger


class Assistant:
    """
    rag assistant for answering questions using web search results
    """

    def __init__(
        self,
        gen_model: str = "phi3:mini",
        vect_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 300,
        chunk_overlap_ratio: float = 0.2,
        logger=logger
    ):
        self.vect_model = SentenceTransformerEmbeddings(model_name=vect_model)
        self.llm = OllamaLLM(
            model=gen_model,
            temperature=0,
            base_url="http://localhost:11434",
        )

        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=
            """ hi, you are an assistant that answers questions about recent news using the provided context from search result.
            If the answer is not in the context, say "I don't know".

            Context:
            {context}

            Question:
            {question}

            Answer (be concise):"""
        )

        self.chunk_size = chunk_size
        self.chunk_overlap = int(chunk_size * chunk_overlap_ratio)
        self.logger = logger

    def get_search_results(self, query, num_res=8, num_reformulations=4):
        """
        returns ddgs web search results for the original query + reformulations
        """
        reformulations = self.generate_web_search_reformulations(query, n=num_reformulations)
        all_results = []
        url_seen = set()

        with DDGS() as ddgs:
            for r in reformulations:
                raw_results = list(ddgs.text(r, max_results=num_res))
                for res in raw_results:
                    content = res.get("body", "")
                    url = res.get("href")
                    if not content.strip() or not url:
                        continue

                    if url not in url_seen:
                        metadata = {"title": res.get("title"), "url": url}
                        all_results.append(Document(page_content=content, metadata=metadata))

        return all_results


    def generate_web_search_reformulations(self, query: str, n: int = 4) -> list:
        """
        returns list of lln reformulated queries for web search
        """
        system = (
            "You are a helpful search query paraphraser. "
            "Given a user's search question, return N short paraphrased web search queries "
            "that cover different angles, keywords, and phrasing. Keep each paraphrase concise."
        )

        prompt = (
            f"{system}\n\nUser query: \"{query}\"\n\n"
            f"Return {n} paraphrased search queries, one per line. Include the original phrasing as the first line."
        )
        resp = self.llm.invoke(prompt)
        lines = [l.strip("-• \t") for l in resp.splitlines() if l.strip()]
        reformulations = []
        seen = set()

        reformulations.append(query)
        seen.add(query.lower())
        for l in lines:
            if l.lower() not in seen:
                reformulations.append(l)
                seen.add(l.lower())
            if len(reformulations) >= n:
                break
        self.logger.info(f"web search reformulations: {reformulations}")
        return reformulations


    def generate_reformulations(self, query: str, n: int = 4) -> list:
        """
        returns list of lln reformulated queries
        """
        system = (
            "You are a helpful search query paraphraser. "
            "Given a user's search question, return N short paraphrased search queries "
            "that cover different angles, keywords, and phrasing. Keep each paraphrase concise."
        )

        prompt = (
            f"{system}\n\nUser query: \"{query}\"\n\n"
            f"Return {n} paraphrased search queries, one per line"
        )
        resp = self.llm.invoke(prompt)

        lines = [l.strip("-• \t") for l in resp.splitlines() if l.strip()]
        reformulations = []
        seen = set()
        reformulations.append(query)
        seen.add(query.lower())

        for l in lines:
            if l.lower() not in seen:
                reformulations.append(l)
                seen.add(l.lower())
            if len(reformulations) >= n:
                break
        self.logger.info(f"reformulations: {reformulations}")
        return reformulations

    def generate_expansions(self, query: str, max_terms: int = 6) -> list:
        """
        generate expansion terms
        """
        prompt = (
            "Given the following search query, return a comma-separated list (no explanation) "
            "of up to {max_terms} high-quality expansion terms or short phrases (synonyms, abbreviations, related terms) "
            "that would help retrieve relevant passages. Keep them short.\n\n"
            "Query: \"{query}\""
        ).format(max_terms=max_terms, query=query)

        resp = self.llm.invoke(prompt)

        items = re.split(r"[,;\n]+", resp)
        items = [it.strip() for it in items if it.strip()]
        unique = []
        seen = set()
        for it in items:
            if it.lower() not in seen:
                unique.append(it)
                seen.add(it.lower())
            if len(unique) >= max_terms:
                break
        self.logger.info(f"expansions terms: {unique}")
        return unique


    @staticmethod
    def term_overlap_score(doc_text: str, query_tokens: list) -> float:
        doc_tokens = re.findall(r"\w+", doc_text.lower())
        if not doc_tokens:
            return 0.0
        doc_set = set(doc_tokens)
        overlap = sum(1 for t in query_tokens if t in doc_set)
        return overlap / math.sqrt(len(doc_tokens))

    def fuse_retrievals(self, retrieval_results: list):
        """
        sort retrievals based on freq and sim_score
        """
        doc_acc = {}
        counts = Counter()
        sum_sim = defaultdict(float)

        for result_list in retrieval_results:
            for doc, sim in result_list:
                key = (doc.metadata.get("url"), doc.metadata.get("title"), doc.page_content[:200])
                counts[key] += 1
                sum_sim[key] += float(sim if sim is not None else 0.0)
                if key not in doc_acc:
                    doc_acc[key] = doc

        vals = list(sum_sim.values())
        vmin, vmax = min(vals), max(vals)
        range_v = vmax - vmin
        norm_sim = {k: (v - vmin) / range_v for k, v in sum_sim.items()}

        fused = []
        for k in doc_acc.keys():
            fused.append({
                "key": k,
                "doc": doc_acc[k],
                "count": counts[k],
                "sim_sum": sum_sim[k],
                "sim_norm": norm_sim.get(k, 0.0)
            })

        fused_sorted = sorted(fused, key=lambda x: (x["count"], x["sim_norm"]), reverse=True)

        return fused_sorted


    def get_best_hits(
        self,
        docs,
        query,
        top_k: int = 3,
        num_refs: int = 4,
        expand_terms: int = 6,
        per_reform_k: int = 6,
        alpha_lexical: float = 0.35,
    ):
        """
        get top_k best docs using reformulations,
        """

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunked_docs = splitter.split_documents(docs)
        vectordb = Chroma.from_documents(chunked_docs, self.vect_model)

        reformulations = self.generate_reformulations(query, n=num_refs)
        all_retrievals = []
        for rq in reformulations:
            results = vectordb.similarity_search_with_score(rq, k=per_reform_k)
            all_retrievals.append(results)

        fused = self.fuse_retrievals(all_retrievals)


        expansions = self.generate_expansions(query, max_terms=expand_terms)
        base_tokens = re.findall(r"\w+", query.lower())
        expansion_tokens = []
        for e in expansions:
            expansion_tokens.extend(re.findall(r"\w+", e.lower()))
        expanded_query_tokens = list(dict.fromkeys(base_tokens + expansion_tokens))  # preserve order, unique


        sim_norms = [f["sim_norm"] for f in fused] if fused else []
        if sim_norms:
            smin, smax = min(sim_norms), max(sim_norms)
            srange = smax - smin if smax > smin else 1.0
        else:
            smin, smax, srange = 0.0, 0.0, 1.0

        scored_candidates = []
        for item in fused:
            doc = item["doc"]
            sim_norm = (item["sim_norm"] - smin) / srange if srange else 0.0
            lexical = self.term_overlap_score(doc.page_content, expanded_query_tokens)
            lex_norm = lexical  # already normalized by sqrt(doc length); keep as-is
            final_score = (1.0 - alpha_lexical) * sim_norm + alpha_lexical * lex_norm
            scored_candidates.append((doc, final_score, {"sim_norm": sim_norm, "lex_norm": lex_norm, "count": item["count"]}))

        scored_sorted = sorted(scored_candidates, key=lambda x: x[1], reverse=True)
        top_docs = [t[0] for t in scored_sorted[:top_k]]

        return top_docs

    def run(self, query, top_n_docs=3):

        docs = self.get_search_results(query)
        best_hits = self.get_best_hits(docs, query, top_k=top_n_docs)
        context_text = "\n\n".join(
            [f"Title: {d.metadata.get('title')}\nURL: {d.metadata.get('url')}\n{d.page_content}" for d in best_hits]
        )
        final_prompt = self.prompt.format(context=context_text, question=query)

        self.logger.info(f"finalprompt: {final_prompt}")
        answer = self.llm.invoke(final_prompt)
        return answer



if __name__ == "__main__":
    assistant = Assistant()
    query = str(input("Ask me:"))
    answer = assistant.run(query)
    print(answer)