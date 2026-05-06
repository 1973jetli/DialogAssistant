from __future__ import annotations

import os
from datetime import datetime
from enum import Enum
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Private Board AI Assistant - Phase 2")


class MeetingState(str, Enum):
    idle = "IDLE"
    running = "RUNNING"
    paused = "PAUSED"
    ended = "ENDED"


class TranscriptEntry(BaseModel):
    speaker: str
    text: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PromptEntry(BaseModel):
    content: str
    severity: str = "normal"
    sources: List[str] = []
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class KnowledgeDoc(BaseModel):
    doc_id: str
    title: str
    source_type: str  # local | link
    source_value: str
    enabled: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MeetingSession(BaseModel):
    topic: str = ""
    selected_docs: List[str] = []
    state: MeetingState = MeetingState.idle
    transcripts: List[TranscriptEntry] = []
    prompts: List[PromptEntry] = []


class StartRequest(BaseModel):
    topic: str
    selected_docs: List[str] = []
    use_web_search: bool = True


class SpeechRequest(BaseModel):
    speaker: str
    text: str


class DocsSelectRequest(BaseModel):
    doc_ids: List[str]


class DocsUpsertRequest(BaseModel):
    title: str
    source_type: str
    source_value: str


class OpenRouterConfig(BaseModel):
    model: str = "qwen/qwen3-32b"
    max_tokens: int = 240
    temperature: float = 0.6


session = MeetingSession()
knowledge_docs: List[KnowledgeDoc] = []
openrouter_config = OpenRouterConfig()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/session")
def get_session() -> MeetingSession:
    return session


@app.post("/api/start")
def start_meeting(payload: StartRequest) -> MeetingSession:
    global session
    invalid = [d for d in payload.selected_docs if d not in {x.doc_id for x in knowledge_docs}]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid doc ids: {invalid}")
    session = MeetingSession(topic=payload.topic, selected_docs=payload.selected_docs, state=MeetingState.running)
    return session


@app.post("/api/pause")
def pause_meeting() -> MeetingSession:
    if session.state != MeetingState.running:
        raise HTTPException(status_code=400, detail="Meeting is not running")
    session.state = MeetingState.paused
    return session


@app.post("/api/restart")
def restart_meeting() -> MeetingSession:
    if session.state != MeetingState.paused:
        raise HTTPException(status_code=400, detail="Meeting is not paused")
    session.state = MeetingState.running
    return session


@app.post("/api/end")
def end_meeting() -> dict:
    if session.state not in {MeetingState.running, MeetingState.paused}:
        raise HTTPException(status_code=400, detail="Meeting has not started")
    session.state = MeetingState.ended
    return {
        "topic": session.topic,
        "core_points": _core_points(session.transcripts),
        "action_items": _extract_action_items([x.text for x in session.transcripts]),
        "mood": _mood(session.transcripts),
        "selected_docs": session.selected_docs,
    }


@app.post("/api/speech")
def add_speech(payload: SpeechRequest) -> dict:
    if session.state != MeetingState.running:
        raise HTTPException(status_code=400, detail="Meeting is not running")

    entry = TranscriptEntry(speaker=payload.speaker, text=payload.text)
    session.transcripts.append(entry)

    prompt = _generate_prompt(payload.text, session.topic, session.selected_docs)
    if prompt:
        session.prompts.append(prompt)

    return {"ok": True, "prompt": prompt}


@app.get("/api/docs")
def list_docs() -> List[KnowledgeDoc]:
    return knowledge_docs


@app.post("/api/docs")
def create_doc(payload: DocsUpsertRequest) -> KnowledgeDoc:
    if payload.source_type not in {"local", "link"}:
        raise HTTPException(status_code=400, detail="source_type must be local or link")
    doc_id = f"doc_{len(knowledge_docs) + 1}"
    doc = KnowledgeDoc(doc_id=doc_id, title=payload.title, source_type=payload.source_type, source_value=payload.source_value)
    knowledge_docs.append(doc)
    return doc


@app.delete("/api/docs/{doc_id}")
def delete_doc(doc_id: str) -> dict:
    global knowledge_docs
    before = len(knowledge_docs)
    knowledge_docs = [d for d in knowledge_docs if d.doc_id != doc_id]
    if len(knowledge_docs) == before:
        raise HTTPException(status_code=404, detail="doc not found")
    if doc_id in session.selected_docs:
        session.selected_docs.remove(doc_id)
    return {"ok": True}


@app.post("/api/docs/select")
def select_docs(payload: DocsSelectRequest) -> dict:
    known = {x.doc_id for x in knowledge_docs}
    invalid = [d for d in payload.doc_ids if d not in known]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid doc ids: {invalid}")
    session.selected_docs = payload.doc_ids
    return {"ok": True, "selected_docs": session.selected_docs}


@app.post("/api/config/openrouter")
def set_openrouter_config(payload: OpenRouterConfig) -> OpenRouterConfig:
    global openrouter_config
    openrouter_config = payload
    return openrouter_config


def _generate_prompt(text: str, topic: str, selected_docs: List[str]) -> Optional[PromptEntry]:
    lower = text.lower()
    if topic and not any(k in lower for k in topic.lower().split() if len(k) > 1):
        return PromptEntry(content=f"⚠️ 当前讨论可能偏离主题“{topic}”，建议回到核心议题。", severity="warning")

    context_snippets = _build_context(selected_docs)
    result = _call_openrouter(topic, text, context_snippets)
    if result:
        return PromptEntry(content=result, severity="normal", sources=context_snippets)

    trigger_words = ["我觉得", "应该", "风险", "成本", "增长", "用户"]
    if any(w in text for w in trigger_words):
        return PromptEntry(content="小虎提问：这个方案忽略了哪些反例或失败场景？", severity="normal")
    return None


def _build_context(selected_docs: List[str]) -> List[str]:
    docs = [d for d in knowledge_docs if d.doc_id in selected_docs]
    snippets = [f"{d.title}: {d.source_value}" for d in docs[:3]]
    return snippets


def _call_openrouter(topic: str, latest_speech: str, snippets: List[str]) -> Optional[str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None

    system_prompt = (
        "你是私董会成员'小虎'，客观中立、善于挑战假设、识别情绪并提出启发性问题。"
        "提问尽量围绕主题；如偏题则指出。回答仅输出一个问题。"
    )
    user_prompt = (
        f"主题: {topic}\n"
        f"最新发言: {latest_speech}\n"
        f"知识库上下文: {snippets}\n"
        "请生成一句高质量挑战性问题。"
    )

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": openrouter_config.model,
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                "temperature": openrouter_config.temperature,
                "max_tokens": openrouter_config.max_tokens,
            },
            timeout=15,
        )
        if not resp.ok:
            return None
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def _core_points(items: List[TranscriptEntry]) -> List[str]:
    if not items:
        return ["暂无讨论内容"]
    return [f"{i.speaker}：{i.text}" for i in items[-8:]]


def _extract_action_items(texts: List[str]) -> List[str]:
    actions = [t for t in texts if any(k in t for k in ["下周", "负责", "行动", "完成", "deadline", "明天"])]
    return actions or ["暂无明确行动项，请补充负责人、截止时间和交付物。"]


def _mood(items: List[TranscriptEntry]) -> str:
    text = " ".join(x.text for x in items)
    if any(k in text for k in ["担心", "焦虑", "风险", "困难", "压力"]):
        return "观察到谨慎或压力情绪，建议先澄清目标与边界。"
    if any(k in text for k in ["兴奋", "机会", "增长", "突破", "期待"]):
        return "观察到积极进取情绪，建议同步评估执行风险。"
    return "情绪整体平稳。"
