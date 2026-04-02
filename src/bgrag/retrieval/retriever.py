"""Hybrid retrieval orchestration."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Sequence

import math

import bgrag.retrieval.topology  # Ensure topology policies are registered.
import cohere
from elasticsearch import Elasticsearch

from bgrag.config import Settings
from bgrag.indexing.elastic import VECTOR_FIELD_NAME, chunk_index_name
from bgrag.registry import source_topology_registry
from bgrag.retrieval.packing import diversify_ranked_chunks
from bgrag.types import ChunkRecord, EvidenceBundle, NormalizedDocument, RetrievalCandidate


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    numerator = sum(x * y for x, y in zip(a, b))
    denom_a = math.sqrt(sum(x * x for x in a))
    denom_b = math.sqrt(sum(y * y for y in b))
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return numerator / (denom_a * denom_b)


def _heading_prefix(chunk: ChunkRecord) -> tuple[str, ...]:
    if len(chunk.heading_path) >= 2:
        return tuple(chunk.heading_path[:-1])
    if chunk.heading_path:
        return tuple(chunk.heading_path)
    if chunk.heading:
        return (chunk.heading,)
    return (chunk.title,)


def _question_terms(question: str) -> set[str]:
    return {
        term
        for term in "".join(char.lower() if char.isalnum() else " " for char in question).split()
        if len(term) >= 4
    }


def _term_overlap_score(question_terms: set[str], text: str) -> int:
    if not question_terms:
        return 0
    text_terms = {
        term
        for term in "".join(char.lower() if char.isalnum() else " " for char in text).split()
        if len(term) >= 4
    }
    return len(question_terms & text_terms)


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        elastic: Elasticsearch | None = None,
        index_namespace: str | None = None,
        documents: list[NormalizedDocument] | None = None,
    ) -> None:
        self.settings = settings
        self.elastic = elastic
        self.index_namespace = index_namespace or "default"
        self.rerank_client = cohere.ClientV2(settings.cohere_api_key) if settings.cohere_api_key else None
        self.documents_by_id = {document.doc_id: document for document in documents or []}
        self.documents_by_url = {document.canonical_url: document for document in documents or []}
        self._chunks_cache_key: int | None = None
        self._chunk_by_id: dict[str, ChunkRecord] = {}
        self._chunks_by_doc: dict[str, list[ChunkRecord]] = {}
        self._intro_chunks_by_order: dict[int, list[ChunkRecord]] = {}

    def _ensure_chunk_views(self, chunks: list[ChunkRecord]) -> None:
        cache_key = id(chunks)
        if self._chunks_cache_key == cache_key:
            return
        grouped: dict[str, list[ChunkRecord]] = defaultdict(list)
        intro_by_order: dict[int, list[ChunkRecord]] = defaultdict(list)
        chunk_by_id: dict[str, ChunkRecord] = {}
        for chunk in chunks:
            grouped[chunk.doc_id].append(chunk)
            chunk_by_id[chunk.chunk_id] = chunk
            if chunk.chunker_name == "section_chunker":
                intro_by_order[chunk.order].append(chunk)
        for doc_chunks in grouped.values():
            doc_chunks.sort(key=lambda item: item.order)
        self._chunks_by_doc = grouped
        self._intro_chunks_by_order = intro_by_order
        self._chunk_by_id = chunk_by_id
        self._chunks_cache_key = cache_key

    def _iter_source_families(
        self,
        chunks: list[ChunkRecord],
        *,
        allowed_chunk_ids: set[str] | None = None,
    ) -> list[str]:
        families: set[str] = set()
        if allowed_chunk_ids is None:
            for chunk in chunks:
                families.add(chunk.source_family.value)
            return sorted(families)
        for chunk in chunks:
            if chunk.chunk_id in allowed_chunk_ids:
                families.add(chunk.source_family.value)
        return sorted(families)

    def lexical_search(
        self,
        question: str,
        chunks: list[ChunkRecord],
        top_k: int,
        *,
        allowed_chunk_ids: set[str] | None = None,
    ) -> dict[str, float]:
        if self.elastic is None:
            raise RuntimeError(
                "Hybrid retrieval requires Elasticsearch-backed lexical search. "
                "Configure Elasticsearch and run `bgrag build-index` before querying."
            )
        scores: dict[str, float] = {}
        found_index = False
        if allowed_chunk_ids is not None and not allowed_chunk_ids:
            return scores
        chunks_by_family: dict[str, set[str]] = defaultdict(set)
        if allowed_chunk_ids is not None:
            for chunk in chunks:
                if chunk.chunk_id in allowed_chunk_ids:
                    chunks_by_family[chunk.source_family.value].add(chunk.chunk_id)
        try:
            for family in self._iter_source_families(chunks, allowed_chunk_ids=allowed_chunk_ids):
                index_name = chunk_index_name(family, self.index_namespace)
                if not self.elastic.indices.exists(index=index_name):
                    continue
                found_index = True
                query: dict[str, object] = {
                    "multi_match": {
                        "query": question,
                        "fields": ["title^2", "heading^2", "heading_path", "text"],
                    }
                }
                if allowed_chunk_ids is not None:
                    family_allowed_ids = sorted(chunks_by_family.get(family, set()))
                    if not family_allowed_ids:
                        continue
                    query = {
                        "bool": {
                            "must": [query],
                            "filter": [{"terms": {"chunk_id": family_allowed_ids}}],
                        }
                    }
                response = self.elastic.search(
                    index=index_name,
                    size=top_k,
                    query=query,
                )
                for hit in response.get("hits", {}).get("hits", []):
                    scores[str(hit["_id"])] = float(hit.get("_score", 0.0))
        except Exception as exc:
            raise RuntimeError(
                "Elasticsearch lexical search failed. "
                "Fix the indexed lexical path instead of relying on a local fallback."
            ) from exc
        if not found_index:
            raise RuntimeError(
                "No Elasticsearch chunk indices were found. "
                "Run `bgrag build-index` before querying or evaluating."
            )
        return scores

    def _dense_scores_from_store(
        self,
        query_embedding: Sequence[float],
        chunks: list[ChunkRecord],
        chunk_embeddings: dict[str, Sequence[float]],
        *,
        allowed_chunk_ids: set[str] | None = None,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for chunk in chunks:
            if allowed_chunk_ids is not None and chunk.chunk_id not in allowed_chunk_ids:
                continue
            vector = chunk_embeddings.get(chunk.chunk_id)
            if vector is None:
                continue
            scores[chunk.chunk_id] = _cosine(query_embedding, vector)
        return scores

    @staticmethod
    def _knn_score_to_cosine(score: float) -> float:
        return max(-1.0, min(1.0, (2.0 * score) - 1.0))

    def vector_search(
        self,
        query_embedding: Sequence[float],
        chunks: list[ChunkRecord],
        top_k: int,
        *,
        num_candidates: int,
        allowed_chunk_ids: set[str] | None = None,
    ) -> dict[str, float]:
        if self.elastic is None:
            raise RuntimeError(
                "Hybrid retrieval requires Elasticsearch-backed vector search. "
                "Configure Elasticsearch and run `bgrag build-index` before querying."
            )
        scores: dict[str, float] = {}
        found_index = False
        if allowed_chunk_ids is not None and not allowed_chunk_ids:
            return scores
        chunks_by_family: dict[str, set[str]] = defaultdict(set)
        if allowed_chunk_ids is not None:
            for chunk in chunks:
                if chunk.chunk_id in allowed_chunk_ids:
                    chunks_by_family[chunk.source_family.value].add(chunk.chunk_id)
        try:
            for family in self._iter_source_families(chunks, allowed_chunk_ids=allowed_chunk_ids):
                index_name = chunk_index_name(family, self.index_namespace)
                if not self.elastic.indices.exists(index=index_name):
                    continue
                found_index = True
                knn_query: dict[str, object] = {
                    "field": VECTOR_FIELD_NAME,
                    "query_vector": list(query_embedding),
                    "k": top_k,
                    "num_candidates": max(num_candidates, top_k),
                }
                if allowed_chunk_ids is not None:
                    family_allowed_ids = sorted(chunks_by_family.get(family, set()))
                    if not family_allowed_ids:
                        continue
                    knn_query["filter"] = {"terms": {"chunk_id": family_allowed_ids}}
                response = self.elastic.search(
                    index=index_name,
                    size=top_k,
                    knn=knn_query,
                    source=False,
                )
                for hit in response.get("hits", {}).get("hits", []):
                    scores[str(hit["_id"])] = self._knn_score_to_cosine(float(hit.get("_score", 0.0)))
        except Exception as exc:
            raise RuntimeError(
                "Elasticsearch vector search failed. "
                "Fix the indexed vector retrieval path instead of falling back silently."
            ) from exc
        if not found_index:
            raise RuntimeError(
                "No Elasticsearch chunk indices were found. "
                "Run `bgrag build-index` before querying or evaluating."
            )
        return scores

    def dense_search(
        self,
        *,
        query_embedding: Sequence[float],
        chunks: list[ChunkRecord],
        chunk_embeddings: dict[str, Sequence[float]],
        top_k: int,
        dense_retrieval_backend: str,
        es_knn_num_candidates: int,
        allowed_chunk_ids: set[str] | None = None,
    ) -> dict[str, float]:
        if dense_retrieval_backend == "local_embedding_store":
            return self._dense_scores_from_store(
                query_embedding,
                chunks,
                chunk_embeddings,
                allowed_chunk_ids=allowed_chunk_ids,
            )
        if dense_retrieval_backend == "elasticsearch_knn":
            return self.vector_search(
                query_embedding,
                chunks,
                top_k=top_k,
                num_candidates=es_knn_num_candidates,
                allowed_chunk_ids=allowed_chunk_ids,
            )
        raise ValueError(f"Unsupported dense_retrieval_backend: {dense_retrieval_backend}")

    def _materialize_candidates(
        self,
        *,
        chunks: list[ChunkRecord],
        dense_scores: dict[str, float],
        lexical_scores: dict[str, float],
        retrieval_alpha: float,
        candidate_k: int,
    ) -> list[RetrievalCandidate]:
        self._ensure_chunk_views(chunks)
        candidate_ids = set(dense_scores) | set(lexical_scores)
        candidates: list[RetrievalCandidate] = []
        for chunk in chunks:
            if chunk.chunk_id not in candidate_ids:
                continue
            dense_score = dense_scores.get(chunk.chunk_id, 0.0)
            lexical_score = lexical_scores.get(chunk.chunk_id, 0.0)
            blended = retrieval_alpha * dense_score + (1.0 - retrieval_alpha) * lexical_score
            candidates.append(
                RetrievalCandidate(
                    chunk=chunk,
                    dense_score=dense_score,
                    lexical_score=lexical_score,
                    blended_score=blended,
                )
            )
        candidates.sort(key=lambda item: item.blended_score, reverse=True)
        return candidates[:candidate_k]

    def _score_query_candidates(
        self,
        *,
        question: str,
        query_embedding: Sequence[float],
        chunks: list[ChunkRecord],
        chunk_embeddings: dict[str, Sequence[float]],
        retrieval_alpha: float,
        candidate_k: int,
        dense_retrieval_backend: str,
        es_knn_num_candidates: int,
        restrict_lexical_search: bool,
    ) -> tuple[list[RetrievalCandidate], dict[str, float]]:
        lexical_kwargs: dict[str, object] = {}
        lexical_top_k = candidate_k
        allowed_chunk_ids: set[str] | None = None
        if restrict_lexical_search:
            allowed_chunk_ids = {chunk.chunk_id for chunk in chunks}
            lexical_kwargs["allowed_chunk_ids"] = allowed_chunk_ids
            lexical_top_k = max(candidate_k, len(chunks))

        lexical_start = perf_counter()
        lexical_scores = self.lexical_search(question, chunks=chunks, top_k=lexical_top_k, **lexical_kwargs)
        lexical_end = perf_counter()
        vector_start = perf_counter()
        dense_scores = self.dense_search(
            query_embedding=query_embedding,
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
            top_k=candidate_k,
            dense_retrieval_backend=dense_retrieval_backend,
            es_knn_num_candidates=es_knn_num_candidates,
            allowed_chunk_ids=allowed_chunk_ids,
        )
        vector_end = perf_counter()
        fusion_start = perf_counter()
        candidates = self._materialize_candidates(
            chunks=chunks,
            dense_scores=dense_scores,
            lexical_scores=lexical_scores,
            retrieval_alpha=retrieval_alpha,
            candidate_k=candidate_k,
        )
        fusion_end = perf_counter()
        return candidates, {
            "lexical_search_seconds": lexical_end - lexical_start,
            "vector_search_seconds": vector_end - vector_start,
            "candidate_fusion_seconds": fusion_end - fusion_start,
        }

    def _build_single_query_candidates(
        self,
        *,
        question: str,
        query_embedding: Sequence[float],
        chunks: list[ChunkRecord],
        chunk_embeddings: dict[str, Sequence[float]],
        retrieval_alpha: float,
        candidate_k: int,
        dense_retrieval_backend: str,
        es_knn_num_candidates: int,
        restrict_lexical_search: bool = False,
        stage_timings: dict[str, float] | None = None,
    ) -> list[RetrievalCandidate]:
        candidates, timings = self._score_query_candidates(
            question=question,
            query_embedding=query_embedding,
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
            retrieval_alpha=retrieval_alpha,
            candidate_k=candidate_k,
            dense_retrieval_backend=dense_retrieval_backend,
            es_knn_num_candidates=es_knn_num_candidates,
            restrict_lexical_search=restrict_lexical_search,
        )
        if stage_timings is not None:
            for key, value in timings.items():
                stage_timings[key] = stage_timings.get(key, 0.0) + value
        return candidates

    def _build_candidate_pool(
        self,
        *,
        question: str,
        retrieval_queries: list[str],
        query_embeddings: list[Sequence[float]],
        chunks: list[ChunkRecord],
        chunk_embeddings: dict[str, Sequence[float]],
        retrieval_alpha: float,
        candidate_k: int,
        per_query_candidate_k: int,
        query_fusion_rrf_k: int,
        dense_retrieval_backend: str,
        es_knn_num_candidates: int,
        restrict_lexical_search: bool = False,
        enable_parallel_query_branches: bool = False,
        stage_timings: dict[str, float] | None = None,
    ) -> list[RetrievalCandidate]:
        if len(retrieval_queries) == 1:
            return self._build_single_query_candidates(
                question=question,
                query_embedding=query_embeddings[0],
                chunks=chunks,
                chunk_embeddings=chunk_embeddings,
                retrieval_alpha=retrieval_alpha,
                candidate_k=candidate_k,
                dense_retrieval_backend=dense_retrieval_backend,
                es_knn_num_candidates=es_knn_num_candidates,
                restrict_lexical_search=restrict_lexical_search,
                stage_timings=stage_timings,
            )
        return self._retrieve_multi_query_candidates(
            retrieval_queries=retrieval_queries,
            query_embeddings=query_embeddings,
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
            retrieval_alpha=retrieval_alpha,
            candidate_k=candidate_k,
            per_query_candidate_k=per_query_candidate_k,
            query_fusion_rrf_k=query_fusion_rrf_k,
            dense_retrieval_backend=dense_retrieval_backend,
            es_knn_num_candidates=es_knn_num_candidates,
            restrict_lexical_search=restrict_lexical_search,
            enable_parallel_query_branches=enable_parallel_query_branches,
            stage_timings=stage_timings,
        )

    def _merge_candidate_groups(
        self,
        groups: list[tuple[str, list[RetrievalCandidate]]],
    ) -> list[RetrievalCandidate]:
        merged: dict[str, RetrievalCandidate] = {}
        for reason, candidates in groups:
            for candidate in candidates:
                existing = merged.get(candidate.chunk.chunk_id)
                if existing is None:
                    merged[candidate.chunk.chunk_id] = RetrievalCandidate(
                        chunk=candidate.chunk,
                        dense_score=candidate.dense_score,
                        lexical_score=candidate.lexical_score,
                        rerank_score=candidate.rerank_score,
                        blended_score=candidate.blended_score,
                        source_topology_reason=reason,
                    )
                    continue
                existing.dense_score = max(existing.dense_score, candidate.dense_score)
                existing.lexical_score = max(existing.lexical_score, candidate.lexical_score)
                existing.rerank_score = max(existing.rerank_score, candidate.rerank_score)
                existing.blended_score = max(existing.blended_score, candidate.blended_score)
                if existing.source_topology_reason is None:
                    existing.source_topology_reason = reason
                elif reason not in existing.source_topology_reason:
                    existing.source_topology_reason = f"{existing.source_topology_reason},{reason}"
        ranked = list(merged.values())
        ranked.sort(key=lambda item: item.blended_score, reverse=True)
        return ranked

    def _intro_chunk_pool(self, chunks: list[ChunkRecord], *, max_order: int) -> list[ChunkRecord]:
        self._ensure_chunk_views(chunks)
        intro_chunks: list[ChunkRecord] = []
        for order in range(max_order + 1):
            intro_chunks.extend(self._intro_chunks_by_order.get(order, []))
        return intro_chunks

    def _document_seed_candidates(
        self,
        *,
        question: str,
        base_candidates: list[RetrievalCandidate],
        chunks: list[ChunkRecord],
        chunk_embeddings: dict[str, Sequence[float]],
        retrieval_queries: list[str],
        query_embeddings: list[Sequence[float]],
        retrieval_alpha: float,
        per_query_candidate_k: int,
        query_fusion_rrf_k: int,
        dense_retrieval_backend: str,
        es_knn_num_candidates: int,
        ranking_mode: str,
        scope: str,
        scope_docs: int,
        seed_docs: int,
        intro_max_order: int,
        intro_chunks_per_doc: int,
        candidate_k: int,
        max_chars: int,
    ) -> list[RetrievalCandidate]:
        self._ensure_chunk_views(chunks)
        question_terms = _question_terms(question)
        seed_doc_ids: list[str] = []
        doc_scores: dict[str, float] = {}

        if ranking_mode == "rerank_docs":
            if self.rerank_client is None:
                return []
            scoped_doc_ids: set[str] | None = None
            if scope in {"local_graph", "local_lineage"}:
                scoped_doc_ids = set()
                anchor_doc_ids: list[str] = []
                seen_anchor_docs: set[str] = set()
                for candidate in base_candidates:
                    doc_id = candidate.chunk.doc_id
                    if doc_id in seen_anchor_docs:
                        continue
                    seen_anchor_docs.add(doc_id)
                    anchor_doc_ids.append(doc_id)
                    if len(anchor_doc_ids) >= scope_docs:
                        break
                for doc_id in anchor_doc_ids:
                    scoped_doc_ids.add(doc_id)
                    document = self.documents_by_id.get(doc_id)
                    if document is None:
                        continue
                    if document.graph.parent_doc_id:
                        scoped_doc_ids.add(document.graph.parent_doc_id)
                        parent_doc = self.documents_by_id.get(document.graph.parent_doc_id)
                        if parent_doc is not None:
                            scoped_doc_ids.update(parent_doc.graph.child_doc_ids)
                    scoped_doc_ids.update(document.graph.child_doc_ids)
                    if scope == "local_graph":
                        for related_url in document.graph.outgoing_in_scope_links + document.graph.incoming_in_scope_links:
                            related_doc = self.documents_by_url.get(related_url)
                            if related_doc is not None:
                                scoped_doc_ids.add(related_doc.doc_id)
            doc_records: list[tuple[str, str]] = []
            for doc_id, document in self.documents_by_id.items():
                if scoped_doc_ids is not None and doc_id not in scoped_doc_ids:
                    continue
                doc_chunks = self._chunks_by_doc.get(doc_id, [])
                if not doc_chunks:
                    continue
                intro_parts: list[str] = []
                for chunk in doc_chunks:
                    if chunk.order > intro_max_order:
                        continue
                    heading = f"{chunk.heading}: " if chunk.heading else ""
                    intro_parts.append(f"{heading}{chunk.text}")
                    if len(" ".join(intro_parts)) >= max_chars:
                        break
                breadcrumb_titles = " > ".join(link.title for link in document.breadcrumbs)
                summary = "\n".join(
                    part
                    for part in [
                        f"Title: {document.title}",
                        f"URL: {document.canonical_url}",
                        f"Breadcrumbs: {breadcrumb_titles}" if breadcrumb_titles else "",
                        f"Intro: {' '.join(intro_parts)[:max_chars]}",
                    ]
                    if part
                )
                doc_records.append((doc_id, summary))
            if not doc_records:
                return []
            response = self.rerank_client.rerank(
                model=self.settings.cohere_rerank_model,
                query=question,
                documents=[summary for _, summary in doc_records],
                top_n=min(max(seed_docs * 2, seed_docs), len(doc_records)),
            )
            for result in response.results:
                doc_id = doc_records[result.index][0]
                if doc_id in doc_scores:
                    continue
                seed_doc_ids.append(doc_id)
                doc_scores[doc_id] = float(result.relevance_score)
                if len(seed_doc_ids) >= seed_docs:
                    break
        else:
            intro_pool = self._intro_chunk_pool(chunks, max_order=intro_max_order)
            if not intro_pool:
                return []
            intro_candidates = self._build_candidate_pool(
                question=question,
                retrieval_queries=retrieval_queries,
                query_embeddings=query_embeddings,
                chunks=intro_pool,
                chunk_embeddings=chunk_embeddings,
                retrieval_alpha=retrieval_alpha,
                candidate_k=max(candidate_k, seed_docs * max(1, intro_chunks_per_doc)),
                per_query_candidate_k=min(per_query_candidate_k, max(candidate_k, seed_docs * max(1, intro_chunks_per_doc))),
                query_fusion_rrf_k=query_fusion_rrf_k,
                dense_retrieval_backend=dense_retrieval_backend,
                es_knn_num_candidates=es_knn_num_candidates,
                restrict_lexical_search=True,
            )
            if not intro_candidates:
                return []

            for candidate in intro_candidates:
                doc_id = candidate.chunk.doc_id
                if doc_id not in doc_scores:
                    seed_doc_ids.append(doc_id)
                    doc_scores[doc_id] = candidate.blended_score
                    if len(seed_doc_ids) >= seed_docs:
                        break
                else:
                    doc_scores[doc_id] = max(doc_scores[doc_id], candidate.blended_score)
        if not seed_doc_ids:
            return []
        self._ensure_chunk_views(chunks)

        augmented: dict[str, RetrievalCandidate] = {}

        def add(chunk: ChunkRecord, *, blended_score: float, reason: str) -> None:
            existing = augmented.get(chunk.chunk_id)
            if existing is None or blended_score > existing.blended_score:
                augmented[chunk.chunk_id] = RetrievalCandidate(
                    chunk=chunk,
                    blended_score=blended_score,
                    source_topology_reason=reason,
                )

        for doc_id in seed_doc_ids:
            doc_chunks = self._chunks_by_doc.get(doc_id, [])
            if not doc_chunks:
                continue
            doc_score = doc_scores.get(doc_id, 0.0)
            intro_chunks = [chunk for chunk in doc_chunks if chunk.order <= intro_max_order]
            intro_chunks.sort(
                key=lambda chunk: (
                    -_term_overlap_score(question_terms, f"{chunk.title} {chunk.heading or ''} {chunk.text}"),
                    chunk.order,
                )
            )
            for index, chunk in enumerate(intro_chunks[:intro_chunks_per_doc]):
                add(
                    chunk,
                    blended_score=doc_score * (0.98 - 0.02 * index),
                    reason="document_seed_intro",
                )

        seeded_doc_chunks: list[ChunkRecord] = []
        for doc_id in seed_doc_ids:
            seeded_doc_chunks.extend(self._chunks_by_doc.get(doc_id, []))
        if seeded_doc_chunks:
            seeded_candidates = self._build_candidate_pool(
                question=question,
                retrieval_queries=retrieval_queries,
                query_embeddings=query_embeddings,
                chunks=seeded_doc_chunks,
                chunk_embeddings=chunk_embeddings,
                retrieval_alpha=retrieval_alpha,
                candidate_k=candidate_k,
                per_query_candidate_k=min(per_query_candidate_k, candidate_k),
                query_fusion_rrf_k=query_fusion_rrf_k,
                dense_retrieval_backend=dense_retrieval_backend,
                es_knn_num_candidates=es_knn_num_candidates,
                restrict_lexical_search=True,
            )
            for candidate in seeded_candidates:
                add(
                    candidate.chunk,
                    blended_score=max(candidate.blended_score, doc_scores.get(candidate.chunk.doc_id, 0.0) * 0.94),
                    reason="document_seed_retrieval",
                )

        ranked = list(augmented.values())
        ranked.sort(key=lambda item: item.blended_score, reverse=True)
        return ranked

    def _expand_document_context_pool(
        self,
        candidates: list[RetrievalCandidate],
        *,
        chunks: list[ChunkRecord],
        seed_docs: int,
        neighbor_docs: int,
    ) -> list[ChunkRecord]:
        self._ensure_chunk_views(chunks)
        seed_doc_ids: list[str] = []
        seen_docs: set[str] = set()
        for candidate in candidates:
            doc_id = candidate.chunk.doc_id
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            seed_doc_ids.append(doc_id)
            if len(seed_doc_ids) >= seed_docs:
                break

        expanded_doc_ids = list(seed_doc_ids)
        for doc_id in seed_doc_ids:
            document = self.documents_by_id.get(doc_id)
            if document is None:
                continue
            neighbors: list[str] = []
            if document.graph.parent_doc_id:
                siblings = [
                    sibling_id
                    for sibling_id in self.documents_by_id[document.graph.parent_doc_id].graph.child_doc_ids
                    if sibling_id != doc_id
                ]
                neighbors.extend(siblings)
            linked_neighbor_ids: list[str] = []
            for related_url in document.graph.outgoing_in_scope_links + document.graph.incoming_in_scope_links:
                related_doc = self.documents_by_url.get(related_url)
                if related_doc is not None and related_doc.doc_id != doc_id:
                    linked_neighbor_ids.append(related_doc.doc_id)
            neighbors.extend(linked_neighbor_ids)
            unique_neighbors: list[str] = []
            neighbor_seen: set[str] = set()
            for neighbor_id in neighbors:
                if neighbor_id in neighbor_seen or neighbor_id in expanded_doc_ids:
                    continue
                neighbor_seen.add(neighbor_id)
                unique_neighbors.append(neighbor_id)
                if len(unique_neighbors) >= neighbor_docs:
                    break
            expanded_doc_ids.extend(unique_neighbors)

        context_chunks: list[ChunkRecord] = []
        for doc_id in expanded_doc_ids:
            context_chunks.extend(self._chunks_by_doc.get(doc_id, []))
        return context_chunks

    def _structural_context_candidates(
        self,
        *,
        question: str,
        candidates: list[RetrievalCandidate],
        chunks: list[ChunkRecord],
        seed_docs: int,
        intro_max_order: int,
        same_heading_k: int,
        nearby_k: int,
        nearby_window: int,
        neighbor_docs: int,
    ) -> list[RetrievalCandidate]:
        if not candidates:
            return []
        self._ensure_chunk_views(chunks)
        question_terms = _question_terms(question)
        seed_doc_ids: list[str] = []
        doc_anchor_scores: dict[str, float] = {}
        for candidate in candidates:
            doc_id = candidate.chunk.doc_id
            if doc_id not in doc_anchor_scores:
                seed_doc_ids.append(doc_id)
                doc_anchor_scores[doc_id] = candidate.blended_score
                if len(seed_doc_ids) >= seed_docs:
                    break
            else:
                doc_anchor_scores[doc_id] = max(doc_anchor_scores[doc_id], candidate.blended_score)

        seed_candidate_counts: Counter[str] = Counter()
        anchor_candidates: list[RetrievalCandidate] = []
        for candidate in candidates:
            doc_id = candidate.chunk.doc_id
            if doc_id not in doc_anchor_scores:
                continue
            if seed_candidate_counts[doc_id] >= 2:
                continue
            anchor_candidates.append(candidate)
            seed_candidate_counts[doc_id] += 1

        existing_ids = {candidate.chunk.chunk_id for candidate in candidates}
        augmented: dict[str, RetrievalCandidate] = {}

        def add(chunk: ChunkRecord, *, blended_score: float, reason: str) -> None:
            if chunk.chunk_id in existing_ids:
                return
            existing = augmented.get(chunk.chunk_id)
            if existing is None or blended_score > existing.blended_score:
                augmented[chunk.chunk_id] = RetrievalCandidate(
                    chunk=chunk,
                    blended_score=blended_score,
                    source_topology_reason=reason,
                )

        for doc_id in seed_doc_ids:
            doc_chunks = self._chunks_by_doc.get(doc_id, [])
            if not doc_chunks:
                continue
            doc_score = doc_anchor_scores.get(doc_id, 0.0)
            intro_chunks = [chunk for chunk in doc_chunks if chunk.order <= intro_max_order]
            intro_chunks.sort(
                key=lambda chunk: (
                    -_term_overlap_score(question_terms, chunk.text),
                    chunk.order,
                )
            )
            for index, chunk in enumerate(intro_chunks[: max(2, min(4, len(intro_chunks)))]):
                add(
                    chunk,
                    blended_score=doc_score * (0.92 - 0.03 * index),
                    reason="structural_doc_intro",
                )

        for candidate in anchor_candidates:
            doc_chunks = self._chunks_by_doc.get(candidate.chunk.doc_id, [])
            if not doc_chunks:
                continue
            heading_prefix = _heading_prefix(candidate.chunk)
            same_heading = [
                chunk
                for chunk in doc_chunks
                if chunk.chunk_id != candidate.chunk.chunk_id and _heading_prefix(chunk) == heading_prefix
            ]
            same_heading.sort(
                key=lambda chunk: (
                    -_term_overlap_score(question_terms, chunk.text),
                    abs(chunk.order - candidate.chunk.order),
                    chunk.order,
                )
            )
            for index, chunk in enumerate(same_heading[:same_heading_k]):
                add(
                    chunk,
                    blended_score=candidate.blended_score * (0.9 - 0.03 * index),
                    reason="structural_same_heading",
                )

            nearby = [
                chunk
                for chunk in doc_chunks
                if chunk.chunk_id != candidate.chunk.chunk_id
                and abs(chunk.order - candidate.chunk.order) <= nearby_window
            ]
            nearby.sort(
                key=lambda chunk: (
                    -_term_overlap_score(question_terms, chunk.text),
                    abs(chunk.order - candidate.chunk.order),
                    chunk.order,
                )
            )
            for index, chunk in enumerate(nearby[:nearby_k]):
                add(
                    chunk,
                    blended_score=candidate.blended_score * (0.87 - 0.02 * index),
                    reason="structural_nearby",
                )

        for doc_id in seed_doc_ids:
            document = self.documents_by_id.get(doc_id)
            if document is None:
                continue
            doc_score = doc_anchor_scores.get(doc_id, 0.0)
            neighbor_ids: list[str] = []
            if document.graph.parent_doc_id and document.graph.parent_doc_id in self.documents_by_id:
                for sibling_id in self.documents_by_id[document.graph.parent_doc_id].graph.child_doc_ids:
                    if sibling_id != doc_id:
                        neighbor_ids.append(sibling_id)
            for related_url in document.graph.outgoing_in_scope_links + document.graph.incoming_in_scope_links:
                related_doc = self.documents_by_url.get(related_url)
                if related_doc is not None and related_doc.doc_id != doc_id:
                    neighbor_ids.append(related_doc.doc_id)

            unique_neighbor_ids: list[str] = []
            seen_neighbor_ids: set[str] = set()
            for neighbor_id in neighbor_ids:
                if neighbor_id in seen_neighbor_ids or neighbor_id in doc_anchor_scores:
                    continue
                seen_neighbor_ids.add(neighbor_id)
                unique_neighbor_ids.append(neighbor_id)
                if len(unique_neighbor_ids) >= neighbor_docs:
                    break

            for neighbor_id in unique_neighbor_ids:
                neighbor_chunks = self._chunks_by_doc.get(neighbor_id, [])
                intro_chunks = [chunk for chunk in neighbor_chunks if chunk.order <= intro_max_order]
                intro_chunks.sort(
                    key=lambda chunk: (
                        -_term_overlap_score(question_terms, chunk.text),
                        chunk.order,
                    )
                )
                for index, chunk in enumerate(intro_chunks[: max(1, min(3, len(intro_chunks)))]):
                    add(
                        chunk,
                        blended_score=doc_score * (0.84 - 0.03 * index),
                        reason="structural_neighbor_intro",
                    )

        ranked = list(augmented.values())
        ranked.sort(key=lambda item: item.blended_score, reverse=True)
        return ranked

    def rerank(self, question: str, candidates: list[RetrievalCandidate], top_n: int) -> list[RetrievalCandidate]:
        if self.rerank_client is None or not candidates:
            return candidates
        documents = [candidate.chunk.text for candidate in candidates[:top_n]]
        response = self.rerank_client.rerank(
            model=self.settings.cohere_rerank_model,
            query=question,
            documents=documents,
            top_n=top_n,
        )
        rerank_scores = {
            candidates[result.index].chunk.chunk_id: float(result.relevance_score)
            for result in response.results
        }
        reranked: list[RetrievalCandidate] = []
        for candidate in candidates:
            candidate.rerank_score = rerank_scores.get(candidate.chunk.chunk_id, 0.0)
            candidate.blended_score = 0.8 * candidate.blended_score + 0.2 * candidate.rerank_score
            reranked.append(candidate)
        reranked.sort(key=lambda item: item.blended_score, reverse=True)
        return reranked

    def mmr_reorder(
        self,
        candidates: list[RetrievalCandidate],
        chunk_embeddings: dict[str, Sequence[float]],
        mmr_lambda: float,
    ) -> list[RetrievalCandidate]:
        if len(candidates) <= 1:
            return candidates
        remaining = list(candidates)
        selected: list[RetrievalCandidate] = []

        while remaining:
            if not selected:
                selected.append(remaining.pop(0))
                continue

            best_index = 0
            best_score = float("-inf")
            for index, candidate in enumerate(remaining):
                candidate_embedding = chunk_embeddings.get(candidate.chunk.chunk_id)
                similarity_penalty = 0.0
                if candidate_embedding is not None:
                    similarity_penalty = max(
                        _cosine(candidate_embedding, chunk_embeddings.get(chosen.chunk.chunk_id, []))
                        for chosen in selected
                    )
                score = mmr_lambda * candidate.blended_score - (1.0 - mmr_lambda) * similarity_penalty
                if score > best_score:
                    best_score = score
                    best_index = index
            selected.append(remaining.pop(best_index))

        return selected

    def retrieve(
        self,
        question: str,
        chunks: list[ChunkRecord],
        query_embedding: Sequence[float] | None,
        chunk_embeddings: dict[str, Sequence[float]] | None,
        source_topology: str,
        top_k: int,
        candidate_k: int,
        retrieval_alpha: float,
        rerank_top_n: int = 0,
        dense_retrieval_backend: str = "local_embedding_store",
        es_knn_num_candidates: int = 120,
        enable_mmr_diversity: bool = False,
        mmr_lambda: float = 0.75,
        enable_ranked_chunk_diversity: bool = False,
        diversity_cover_fraction: float = 0.5,
        max_chunks_per_document: int = 8,
        max_chunks_per_heading: int = 4,
        seed_chunks_per_heading: int = 2,
        retrieval_queries: list[str] | None = None,
        query_embeddings: list[Sequence[float]] | None = None,
        query_fusion_rrf_k: int = 60,
        per_query_candidate_k: int = 24,
        enable_parallel_query_branches: bool = False,
        enable_page_intro_expansion: bool = False,
        page_intro_candidate_k: int = 8,
        page_intro_max_order: int = 10,
        enable_document_context_expansion: bool = False,
        document_context_seed_docs: int = 2,
        document_context_candidate_k: int = 12,
        document_context_neighbor_docs: int = 2,
        enable_structural_context_augmentation: bool = False,
        structural_context_seed_docs: int = 2,
        structural_context_intro_max_order: int = 10,
        structural_context_same_heading_k: int = 2,
        structural_context_nearby_k: int = 3,
        structural_context_nearby_window: int = 12,
        structural_context_neighbor_docs: int = 2,
        enable_document_seed_retrieval: bool = False,
        document_seed_ranking_mode: str = "intro_pool",
        document_seed_scope: str = "corpus",
        document_seed_scope_docs: int = 4,
        document_seed_docs: int = 3,
        document_seed_intro_max_order: int = 10,
        document_seed_intro_chunks: int = 3,
        document_seed_candidate_k: int = 12,
        document_seed_max_chars: int = 1400,
    ) -> EvidenceBundle:
        if query_embedding is None:
            raise RuntimeError(
                "Hybrid retrieval requires a query embedding. "
                "Generate queries with Cohere embeddings instead of degrading to lexical-only retrieval."
            )
        if not chunk_embeddings:
            raise RuntimeError(
                "Hybrid retrieval requires a populated chunk embedding store. "
                "Run `bgrag build-index` with Cohere embeddings before querying."
            )
        active_queries = retrieval_queries or [question]
        active_embeddings = query_embeddings or ([query_embedding] if query_embedding is not None else [])
        if len(active_queries) != len(active_embeddings):
            raise ValueError("retrieval_queries and query_embeddings must have matching lengths")
        stage_timings: dict[str, float] = {
            "lexical_search_seconds": 0.0,
            "vector_search_seconds": 0.0,
            "candidate_fusion_seconds": 0.0,
            "rerank_seconds": 0.0,
            "packing_seconds": 0.0,
        }

        base_candidates = self._build_candidate_pool(
            question=question,
            retrieval_queries=active_queries,
            query_embeddings=active_embeddings,
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
            retrieval_alpha=retrieval_alpha,
            candidate_k=candidate_k,
            per_query_candidate_k=per_query_candidate_k,
            query_fusion_rrf_k=query_fusion_rrf_k,
            dense_retrieval_backend=dense_retrieval_backend,
            es_knn_num_candidates=es_knn_num_candidates,
            restrict_lexical_search=False,
            enable_parallel_query_branches=enable_parallel_query_branches,
            stage_timings=stage_timings,
        )
        candidate_groups: list[tuple[str, list[RetrievalCandidate]]] = [("base", base_candidates)]
        notes: list[str] = []
        if enable_page_intro_expansion:
            intro_chunks = self._intro_chunk_pool(chunks, max_order=page_intro_max_order)
            if intro_chunks:
                intro_candidates = self._build_candidate_pool(
                    question=question,
                    retrieval_queries=active_queries,
                    query_embeddings=active_embeddings,
                    chunks=intro_chunks,
                    chunk_embeddings=chunk_embeddings,
                    retrieval_alpha=retrieval_alpha,
                    candidate_k=page_intro_candidate_k,
                    per_query_candidate_k=min(per_query_candidate_k, page_intro_candidate_k),
                    query_fusion_rrf_k=query_fusion_rrf_k,
                    dense_retrieval_backend=dense_retrieval_backend,
                    es_knn_num_candidates=es_knn_num_candidates,
                    restrict_lexical_search=True,
                    enable_parallel_query_branches=enable_parallel_query_branches,
                    stage_timings=stage_timings,
                )
                if intro_candidates:
                    candidate_groups.append(("page_intro_expansion", intro_candidates))
                    notes.append("page_intro_expansion_applied")
        if enable_document_seed_retrieval and self.documents_by_id:
            document_seed_candidates = self._document_seed_candidates(
                question=question,
                base_candidates=base_candidates,
                retrieval_queries=active_queries,
                query_embeddings=active_embeddings,
                chunks=chunks,
                chunk_embeddings=chunk_embeddings,
                retrieval_alpha=retrieval_alpha,
                per_query_candidate_k=per_query_candidate_k,
                query_fusion_rrf_k=query_fusion_rrf_k,
                dense_retrieval_backend=dense_retrieval_backend,
                es_knn_num_candidates=es_knn_num_candidates,
                ranking_mode=document_seed_ranking_mode,
                scope=document_seed_scope,
                scope_docs=document_seed_scope_docs,
                seed_docs=document_seed_docs,
                intro_max_order=document_seed_intro_max_order,
                intro_chunks_per_doc=document_seed_intro_chunks,
                candidate_k=document_seed_candidate_k,
                max_chars=document_seed_max_chars,
            )
            if document_seed_candidates:
                candidate_groups.append(("document_seed_retrieval", document_seed_candidates))
                notes.append("document_seed_retrieval_applied")
        merged_candidates = self._merge_candidate_groups(candidate_groups)
        if enable_document_context_expansion and self.documents_by_id:
            context_chunks = self._expand_document_context_pool(
                merged_candidates,
                chunks=chunks,
                seed_docs=document_context_seed_docs,
                neighbor_docs=document_context_neighbor_docs,
            )
            if context_chunks:
                context_candidates = self._build_candidate_pool(
                    question=question,
                    retrieval_queries=active_queries,
                    query_embeddings=active_embeddings,
                    chunks=context_chunks,
                    chunk_embeddings=chunk_embeddings,
                    retrieval_alpha=retrieval_alpha,
                    candidate_k=document_context_candidate_k,
                    per_query_candidate_k=min(per_query_candidate_k, document_context_candidate_k),
                    query_fusion_rrf_k=query_fusion_rrf_k,
                    dense_retrieval_backend=dense_retrieval_backend,
                    es_knn_num_candidates=es_knn_num_candidates,
                    restrict_lexical_search=True,
                    enable_parallel_query_branches=enable_parallel_query_branches,
                    stage_timings=stage_timings,
                )
                if context_candidates:
                    merged_candidates = self._merge_candidate_groups(
                        [("merged", merged_candidates), ("document_context_expansion", context_candidates)]
                    )
                    notes.append("document_context_expansion_applied")
        if enable_structural_context_augmentation:
            structural_candidates = self._structural_context_candidates(
                question=question,
                candidates=merged_candidates,
                chunks=chunks,
                seed_docs=structural_context_seed_docs,
                intro_max_order=structural_context_intro_max_order,
                same_heading_k=structural_context_same_heading_k,
                nearby_k=structural_context_nearby_k,
                nearby_window=structural_context_nearby_window,
                neighbor_docs=structural_context_neighbor_docs,
            )
            if structural_candidates:
                merged_candidates = self._merge_candidate_groups(
                    [("merged", merged_candidates), ("structural_context_augmentation", structural_candidates)]
                )
                notes.append("structural_context_augmentation_applied")
        candidates = merged_candidates
        rerank_limit = rerank_top_n or min(len(candidates), candidate_k)
        rerank_start = perf_counter()
        candidates = self.rerank(question, candidates, rerank_limit)
        rerank_end = perf_counter()
        stage_timings["rerank_seconds"] += rerank_end - rerank_start
        if enable_mmr_diversity:
            candidates = self.mmr_reorder(candidates, chunk_embeddings, mmr_lambda)

        packing_start = perf_counter()
        grouped: dict[str, list[ChunkRecord]] = defaultdict(list)
        for candidate in candidates:
            grouped[candidate.chunk.source_family.value].append(candidate.chunk)
        if enable_ranked_chunk_diversity:
            grouped = {
                family: diversify_ranked_chunks(
                    family_chunks,
                    target_k=top_k,
                    cover_fraction=diversity_cover_fraction,
                    max_per_document=max_chunks_per_document,
                    max_per_heading=max_chunks_per_heading,
                    seed_chunks_per_heading=seed_chunks_per_heading,
                )
                for family, family_chunks in grouped.items()
            }
        selected_chunks = source_topology_registry.get(source_topology)(question, grouped, top_k)
        selected_ids = {chunk.chunk_id for chunk in selected_chunks}
        selected_candidates = [candidate for candidate in candidates if candidate.chunk.chunk_id in selected_ids]
        selected_candidates.sort(key=lambda item: item.blended_score, reverse=True)
        packing_end = perf_counter()
        stage_timings["packing_seconds"] += packing_end - packing_start
        if len(active_queries) > 1:
            notes.append("llm_query_decomposition_applied")
        return EvidenceBundle(
            query=question,
            candidates=selected_candidates,
            packed_chunks=selected_chunks,
            retrieval_queries=list(active_queries),
            notes=notes,
            timings=stage_timings,
        )

    def _retrieve_multi_query_candidates(
        self,
        *,
        retrieval_queries: list[str],
        query_embeddings: list[Sequence[float]],
        chunks: list[ChunkRecord],
        chunk_embeddings: dict[str, Sequence[float]],
        retrieval_alpha: float,
        candidate_k: int,
        per_query_candidate_k: int,
        query_fusion_rrf_k: int,
        dense_retrieval_backend: str,
        es_knn_num_candidates: int,
        restrict_lexical_search: bool = False,
        enable_parallel_query_branches: bool = False,
        stage_timings: dict[str, float] | None = None,
    ) -> list[RetrievalCandidate]:
        candidate_limit = max(1, min(candidate_k, per_query_candidate_k))
        fused: dict[str, RetrievalCandidate] = {}
        query_results: list[tuple[list[RetrievalCandidate], dict[str, float]]] = []
        if enable_parallel_query_branches and len(retrieval_queries) > 1:
            max_workers = min(len(retrieval_queries), 4)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self._score_query_candidates,
                        question=query,
                        query_embedding=query_embedding,
                        chunks=chunks,
                        chunk_embeddings=chunk_embeddings,
                        retrieval_alpha=retrieval_alpha,
                        candidate_k=candidate_limit,
                        dense_retrieval_backend=dense_retrieval_backend,
                        es_knn_num_candidates=es_knn_num_candidates,
                        restrict_lexical_search=restrict_lexical_search,
                    )
                    for query, query_embedding in zip(retrieval_queries, query_embeddings)
                ]
                for future in futures:
                    query_results.append(future.result())
        else:
            for query, query_embedding in zip(retrieval_queries, query_embeddings):
                query_results.append(
                    self._score_query_candidates(
                        question=query,
                        query_embedding=query_embedding,
                        chunks=chunks,
                        chunk_embeddings=chunk_embeddings,
                        retrieval_alpha=retrieval_alpha,
                        candidate_k=candidate_limit,
                        dense_retrieval_backend=dense_retrieval_backend,
                        es_knn_num_candidates=es_knn_num_candidates,
                        restrict_lexical_search=restrict_lexical_search,
                    )
                )
        if stage_timings is not None:
            for _, query_timings in query_results:
                for key, value in query_timings.items():
                    stage_timings[key] = stage_timings.get(key, 0.0) + value
        fusion_start = perf_counter()
        for query_candidates, _ in query_results:
            for rank, candidate in enumerate(query_candidates[:candidate_limit], start=1):
                existing = fused.get(candidate.chunk.chunk_id)
                fusion_score = 1.0 / (query_fusion_rrf_k + rank)
                if existing is None:
                    fused[candidate.chunk.chunk_id] = RetrievalCandidate(
                        chunk=candidate.chunk,
                        dense_score=candidate.dense_score,
                        lexical_score=candidate.lexical_score,
                        blended_score=fusion_score,
                    )
                else:
                    existing.blended_score += fusion_score
                    existing.dense_score = max(existing.dense_score, candidate.dense_score)
                    existing.lexical_score = max(existing.lexical_score, candidate.lexical_score)
        fusion_end = perf_counter()
        if stage_timings is not None:
            stage_timings["candidate_fusion_seconds"] = stage_timings.get("candidate_fusion_seconds", 0.0) + (
                fusion_end - fusion_start
            )
        candidates = list(fused.values())
        candidates.sort(key=lambda item: item.blended_score, reverse=True)
        return candidates[:candidate_k]
