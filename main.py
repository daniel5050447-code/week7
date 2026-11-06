"""
Advanced RAG 서버 (FastAPI)
===========================
파이프라인:
  PDF → 청킹 → 임베딩 → FAISS 인덱스(+BM25 인덱스)                      [/build]
  질의 → Multi-Query 확장 → (FAISS Dense + BM25 Sparse) 하이브리드 검색
       → RRF(Reciprocal Rank Fusion) 순위 융합 → 상위 문서 → LLM 답변      [/query]

실시간 상태 전달:
  - 모든 작업(build/query)은 백그라운드 job 으로 실행되고, StreamHandler 가 이벤트를 쌓는다.
  - 클라이언트(GAS)는 GET /events/{job_id}?cursor=N&wait=S 로 롱폴링하여 증분 이벤트를 받는다.
    (GAS UrlFetchApp 은 SSE/청크 스트림을 읽을 수 없어서 '롱폴링'으로 스트리밍을 구현)

주의: 인덱스/Job 상태는 인스턴스 메모리에 있으므로 Cloud Run 은 --max-instances=1 로 배포할 것.

환경변수:
  GEMINI_API_KEY (필수) / API_KEY (선택, GAS와 공유하는 접근키)
  LLM_MODEL, EMBED_MODEL, EMBED_DIM, INDEX_TYPE(flat|hnsw), CHUNK_SIZE, CHUNK_OVERLAP, RRF_K
"""
import io
import os
import re
import time
import uuid
import json
import logging
import threading
from typing import Dict, List, Optional

import numpy as np
import faiss
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Header, HTTPException, Depends
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ───────────────────────── 설정 ─────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
API_KEY = os.getenv("API_KEY", "")                       # 비어 있으면 인증 생략
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
INDEX_TYPE = os.getenv("INDEX_TYPE", "flat").lower()     # flat: IndexFlatL2 / hnsw: IndexHNSWFlat
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))         # 청크 글자 수
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
RRF_K = int(os.getenv("RRF_K", "60"))                    # RRF 상수 (원 논문 기본값 60)
EMBED_BATCH = 50                                         # 임베딩 API 1회 호출당 문서 수
PER_PATH_K = 20                                          # 검색 경로(쿼리×Dense/Sparse)당 가져올 후보 수
MAX_PDF_MB = 30

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag")


# ───────────────────────── 예외/이벤트 ─────────────────────────
class StageError(Exception):
    """어느 단계(stage)에서 왜 실패했는지 정확히 전달하기 위한 예외."""

    def __init__(self, stage: str, message: str, hint: str = ""):
        super().__init__(message)
        self.stage, self.message, self.hint = stage, message, hint


class StreamHandler:
    """
    Job 하나의 이벤트 스트림 버퍼.
    emit()로 쌓고, wait_since()로 클라이언트가 cursor 이후 이벤트를 롱폴링한다.
    이벤트 type: status | progress | queries | retrieval | token | complexity | error | done
    """

    def __init__(self, kind: str):
        self.kind = kind
        self.events: List[dict] = []
        self.done = False
        self.created = time.time()
        self._cond = threading.Condition()

    def emit(self, etype: str, **data):
        with self._cond:
            self.events.append({"i": len(self.events), "ts": round(time.time(), 3), "type": etype, **data})
            if etype in ("done", "error"):
                self.done = True
            self._cond.notify_all()

    def wait_since(self, cursor: int, wait: float) -> dict:
        """cursor 이후 새 이벤트가 생길 때까지 최대 wait 초 대기."""
        with self._cond:
            if len(self.events) <= cursor and not self.done:
                self._cond.wait(timeout=wait)
            return {"events": self.events[cursor:], "cursor": len(self.events), "done": self.done}


JOBS: Dict[str, StreamHandler] = {}


def new_job(kind: str) -> (str, StreamHandler):
    # 1시간 지난 Job 정리 (메모리 누수 방지)
    for jid in [j for j, h in JOBS.items() if time.time() - h.created > 3600]:
        JOBS.pop(jid, None)
    jid = uuid.uuid4().hex[:12]
    JOBS[jid] = StreamHandler(kind)
    return jid, JOBS[jid]


# ───────────────────────── 저장소(인덱스 상태) ─────────────────────────
class Store:
    """현재 활성화된 RAG 인덱스. build 완료 시 통째로 교체(원자적)."""
    chunks: List[dict] = []          # [{id, text, source, page}]
    index = None                      # faiss index
    bm25: Optional[BM25Okapi] = None
    source: str = ""
    dim: int = 0
    built_at: float = 0.0


STORE = Store()
_client: Optional[genai.Client] = None


def get_client() -> genai.Client:
    global _client
    if not GEMINI_API_KEY:
        raise StageError("config", "환경변수 GEMINI_API_KEY 가 설정되지 않았습니다.",
                         "Cloud Run 배포 시 --set-env-vars GEMINI_API_KEY=... 를 지정하세요.")
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def retry(fn, tries=4, base=1.5):
    """429/5xx 등 일시 오류용 지수 백오프 재시도."""
    for n in range(tries):
        try:
            return fn()
        except Exception:
            if n == tries - 1:
                raise
            time.sleep(base ** (n + 1))


# ───────────────────────── 텍스트 처리 ─────────────────────────
_TOKEN_RE = re.compile(r"[0-9A-Za-z]+|[가-힣]+")


def tokenize(text: str) -> List[str]:
    """BM25용 토크나이저. 영문/숫자는 단어 단위, 한글은 어절 + 2-gram(조사 변형 대응)."""
    toks: List[str] = []
    for w in _TOKEN_RE.findall(text.lower()):
        toks.append(w)
        if "가" <= w[0] <= "힣" and len(w) >= 2:
            toks.extend(w[i:i + 2] for i in range(len(w) - 1))
    return toks


def extract_chunks(pdf_bytes: bytes, filename: str) -> List[dict]:
    """PDF → 페이지별 텍스트 → CHUNK_SIZE/OVERLAP 슬라이딩 윈도우 청크. 복잡도 O(총 글자수)."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as e:
        raise StageError("pdf_parse", f"PDF를 열 수 없습니다: {type(e).__name__}: {e}",
                         "손상되었거나 암호가 걸린 PDF일 수 있습니다.")
    chunks, step = [], max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    for pno, page in enumerate(reader.pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        for s in range(0, len(text), step):
            piece = text[s:s + CHUNK_SIZE].strip()
            if len(piece) >= 30:                         # 너무 짧은 조각은 제외
                chunks.append({"id": len(chunks), "text": piece, "source": filename, "page": pno})
            if s + CHUNK_SIZE >= len(text):
                break
    if not chunks:
        raise StageError("pdf_parse", "추출된 텍스트가 없습니다.", "스캔(이미지) PDF라면 OCR이 필요합니다.")
    return chunks


# ───────────────────────── 임베딩 / LLM ─────────────────────────
def embed(texts: List[str], task: str, on_batch=None) -> np.ndarray:
    """텍스트 → L2 정규화된 float32 벡터. task: RETRIEVAL_DOCUMENT | RETRIEVAL_QUERY."""
    client, out = get_client(), []
    for b in range(0, len(texts), EMBED_BATCH):
        batch = texts[b:b + EMBED_BATCH]
        try:
            res = retry(lambda: client.models.embed_content(
                model=EMBED_MODEL, contents=batch,
                config=types.EmbedContentConfig(task_type=task, output_dimensionality=EMBED_DIM)))
        except Exception as e:
            raise StageError("embedding", f"임베딩 API 실패({EMBED_MODEL}): {type(e).__name__}: {e}")
        out.extend(e.values for e in res.embeddings)
        if on_batch:
            on_batch(min(b + EMBED_BATCH, len(texts)), len(texts))
    arr = np.asarray(out, dtype="float32")
    faiss.normalize_L2(arr)     # 정규화하면 L2 거리 순서 == 코사인 유사도 순서
    return arr


def make_index(dim: int):
    """FAISS 인덱스 생성. flat: 정확 검색 O(N·d) / hnsw: 근사 검색 O(log N · d)."""
    if INDEX_TYPE == "hnsw":
        idx = faiss.IndexHNSWFlat(dim, 32)       # M=32
        idx.hnsw.efConstruction = 200
        idx.hnsw.efSearch = 64
        return idx
    return faiss.IndexFlatL2(dim)


def expand_queries(query: str, n: int) -> List[str]:
    """Multi-Query: LLM이 서로 다른 관점(동의어/상위개념/세부질문/키워드형)의 파생 쿼리 n개 생성."""
    if n <= 0:
        return []
    prompt = (
        f"다음 질문으로 문서 검색을 하려 한다. 검색 누락을 줄이기 위해 서로 다른 관점의 검색 쿼리 {n}개를 만들어라.\n"
        "관점 예: 동의어/다른 표현, 더 구체적인 세부 질문, 핵심 키워드 나열형. 원 질문과 같은 언어로 작성.\n"
        f'JSON 문자열 배열만 출력. 예: ["쿼리1","쿼리2"]\n\n질문: {query}'
    )
    try:
        res = retry(lambda: get_client().models.generate_content(
            model=LLM_MODEL, contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.7)))
        arr = json.loads(res.text)
        qs = [q.strip() for q in arr if isinstance(q, str) and q.strip() and q.strip() != query]
        return qs[:n]
    except StageError:
        raise
    except Exception as e:
        log.warning("multi-query 실패, 원 쿼리만 사용: %s", e)
        return []                                 # 확장 실패 시 원 쿼리로 degrade (answer는 계속 진행)


# ───────────────────────── 검색 / RRF ─────────────────────────
def rrf_fuse(ranked_lists: List[tuple], k: int = RRF_K) -> List[tuple]:
    """
    RRF: score(d) = Σ_path 1 / (k + rank_path(d))
    ranked_lists: [(경로이름, [doc_id, ...순위순]), ...]
    반환: [(doc_id, score, [{path, rank}, ...]), ...] 점수 내림차순.  복잡도 O(M log M), M=총 후보 수
    """
    scores: Dict[int, float] = {}
    hits: Dict[int, list] = {}
    for path, ids in ranked_lists:
        for rank, did in enumerate(ids, start=1):
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank)
            hits.setdefault(did, []).append({"path": path, "rank": rank})
    return [(d, s, hits[d]) for d, s in sorted(scores.items(), key=lambda x: -x[1])]


def hybrid_search(queries: List[str], top_k: int) -> List[dict]:
    """쿼리마다 Dense(FAISS)+Sparse(BM25) 검색 → RRF 융합 → 상위 top_k."""
    s = STORE
    qvecs = embed(queries, "RETRIEVAL_QUERY")
    k = min(PER_PATH_K, len(s.chunks))
    _, dense_ids = s.index.search(qvecs, k)                     # (Q, k)
    ranked = []
    for qi, q in enumerate(queries):
        tag = "원본" if qi == 0 else f"확장{qi}"
        ranked.append((f"{tag}/Dense", [int(i) for i in dense_ids[qi] if i >= 0]))
        sc = s.bm25.get_scores(tokenize(q))
        top = np.argsort(-sc)[:k]
        ranked.append((f"{tag}/BM25", [int(i) for i in top if sc[i] > 0]))
    fused = rrf_fuse([r for r in ranked if r[1]])[:top_k]
    return [{
        "rank": n, "rrf_score": round(sc, 5), "source": s.chunks[d]["source"], "page": s.chunks[d]["page"],
        "chunk_id": d, "snippet": s.chunks[d]["text"], "hits": hits,
    } for n, (d, sc, hits) in enumerate(fused, start=1)]


# ───────────────────────── Job: 인덱스 생성 ─────────────────────────
def run_build(h: StreamHandler, pdf_bytes: bytes, filename: str):
    t0 = time.time()
    try:
        h.emit("status", stage="pdf_parse", message=f"PDF 파싱 중: {filename}")
        chunks = extract_chunks(pdf_bytes, filename)
        h.emit("progress", stage="chunk", done=len(chunks), total=len(chunks), pct=15,
               message=f"청킹 완료: {len(chunks)}개 (크기 {CHUNK_SIZE}/겹침 {CHUNK_OVERLAP})")

        def on_batch(done, total):
            h.emit("progress", stage="embedding", done=done, total=total, pct=15 + int(65 * done / total),
                   message=f"임베딩 {done}/{total}")

        vecs = embed([c["text"] for c in chunks], "RETRIEVAL_DOCUMENT", on_batch)

        h.emit("status", stage="faiss", message=f"FAISS {INDEX_TYPE.upper()} 인덱스 구축 중")
        idx = make_index(vecs.shape[1])
        idx.add(vecs)

        h.emit("status", stage="bm25", message="BM25 키워드 인덱스 구축 중")
        bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

        s = Store()
        s.chunks, s.index, s.bm25, s.source, s.dim, s.built_at = chunks, idx, bm25, filename, vecs.shape[1], time.time()
        global STORE
        STORE = s                                  # 완성된 객체로 한 번에 교체

        n, d = len(chunks), vecs.shape[1]
        h.emit("complexity", scope="build", items=[
            {"name": "청킹", "big_o": "O(L)", "detail": f"L=총 글자수, 청크 {n}개"},
            {"name": "임베딩", "big_o": "O(N·d) + API", "detail": f"N={n}, d={d}, API {-(-n // EMBED_BATCH)}회 호출"},
            {"name": "FAISS add", "big_o": "O(N·d)" if INDEX_TYPE == "flat" else "O(N·log N·d)",
             "detail": f"{INDEX_TYPE.upper()} 인덱스"},
            {"name": "BM25 구축", "big_o": "O(T)", "detail": "T=전체 토큰 수"},
        ])
        h.emit("done", message=f"완료: 청크 {n}개 / dim {d} / {time.time() - t0:.1f}s", chunks=n,
               elapsed=round(time.time() - t0, 2))
    except StageError as e:
        h.emit("error", stage=e.stage, message=e.message, hint=e.hint)
    except Exception as e:
        log.exception("build 실패")
        h.emit("error", stage="unknown", message=f"{type(e).__name__}: {e}")


# ───────────────────────── Job: 질의 ─────────────────────────
def run_query(h: StreamHandler, query: str, top_k: int, n_queries: int):
    t0 = time.time()
    try:
        if STORE.index is None:
            raise StageError("no_index", "벡터 DB가 아직 생성되지 않았습니다.", "먼저 'PDF 업로드 → 벡터 DB 생성'을 실행하세요.")

        h.emit("status", stage="multi_query", message=f"Multi-Query 확장 중 (파생 {n_queries}개)")
        queries = [query] + expand_queries(query, n_queries)
        h.emit("queries", queries=queries)

        h.emit("status", stage="search", message=f"하이브리드 검색 + RRF ({len(queries)}쿼리 × Dense/BM25)")
        t1 = time.time()
        docs = hybrid_search(queries, top_k)
        h.emit("retrieval", docs=docs, search_ms=round((time.time() - t1) * 1000))
        if not docs:
            raise StageError("search", "검색 결과가 없습니다.", "질의를 바꿔 보세요.")

        h.emit("status", stage="generate", message="LLM 답변 생성 중")
        ctx = "\n\n".join(f"[{d['rank']}] ({d['source']} p.{d['page']})\n{d['snippet']}" for d in docs)
        prompt = ("아래 [문서]만 근거로 질문에 답하라. 근거가 없으면 '문서에서 찾을 수 없습니다'라고 답하라.\n"
                  "주장마다 근거 문서 번호를 [1], [2] 형태로 표기하라.\n\n"
                  f"[문서]\n{ctx}\n\n[질문]\n{query}")
        try:
            for ch in get_client().models.generate_content_stream(model=LLM_MODEL, contents=prompt):
                if ch.text:
                    h.emit("token", text=ch.text)           # 토큰 단위 스트리밍
        except StageError:
            raise
        except Exception as e:
            raise StageError("generate", f"LLM 호출 실패({LLM_MODEL}): {type(e).__name__}: {e}")

        N, d, Q = len(STORE.chunks), STORE.dim, len(queries)
        M = Q * 2 * min(PER_PATH_K, N)
        h.emit("complexity", scope="query", items=[
            {"name": "Multi-Query", "big_o": "O(1) LLM 호출", "detail": f"쿼리 {Q}개"},
            {"name": "Dense(FAISS)", "big_o": f"O(Q·N·d)" if INDEX_TYPE == "flat" else "O(Q·log N·d)",
             "detail": f"Q={Q}, N={N}, d={d}"},
            {"name": "Sparse(BM25)", "big_o": "O(Q·|q|·N)", "detail": f"Q={Q}, N={N}"},
            {"name": "RRF", "big_o": "O(M log M)", "detail": f"M≈{M}"},
        ])
        h.emit("done", message=f"완료 ({time.time() - t0:.1f}s)", elapsed=round(time.time() - t0, 2))
    except StageError as e:
        h.emit("error", stage=e.stage, message=e.message, hint=e.hint)
    except Exception as e:
        log.exception("query 실패")
        h.emit("error", stage="unknown", message=f"{type(e).__name__}: {e}")


# ───────────────────────── API ─────────────────────────
app = FastAPI(title="Advanced RAG Server")


def check_key(x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "X-API-Key 가 올바르지 않습니다. (Script Properties 의 API_KEY 확인)")


class QueryReq(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    n_queries: int = Field(3, ge=0, le=6)


@app.get("/health")
def health():
    """서버/인덱스 상태 (GAS 연동 상태 표시용). 인증 불필요."""
    return {"ok": True, "index_ready": STORE.index is not None, "source": STORE.source,
            "chunks": len(STORE.chunks), "index_type": INDEX_TYPE.upper(), "dim": STORE.dim,
            "llm": LLM_MODEL, "embed": EMBED_MODEL, "api_key_set": bool(GEMINI_API_KEY)}


@app.post("/build", dependencies=[Depends(check_key)])
async def build(bg: BackgroundTasks, file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_PDF_MB * 1024 * 1024:
        raise HTTPException(413, f"PDF가 {MAX_PDF_MB}MB 를 초과합니다.")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "PDF 형식이 아닙니다. (헤더 %PDF 없음)")
    jid, h = new_job("build")
    bg.add_task(run_build, h, data, file.filename or "document.pdf")
    return {"job_id": jid}


@app.post("/query", dependencies=[Depends(check_key)])
def query(req: QueryReq, bg: BackgroundTasks):
    jid, h = new_job("query")
    bg.add_task(run_query, h, req.query.strip(), req.top_k, req.n_queries)
    return {"job_id": jid}


@app.get("/events/{job_id}", dependencies=[Depends(check_key)])
def events(job_id: str, cursor: int = 0, wait: float = 15.0):
    """롱폴링: cursor 이후 이벤트 반환. 없으면 최대 wait초 대기."""
    h = JOBS.get(job_id)
    if h is None:
        raise HTTPException(404, f"job_id={job_id} 없음 (만료되었거나 인스턴스가 재시작됨)")
    return h.wait_since(max(0, cursor), min(max(wait, 0), 25))
