import datetime as dt
import hashlib
import io
import json
import random
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import (JSON, Column, Date, DateTime, Float, Integer, MetaData,
                        String, Table, UniqueConstraint, create_engine, func,
                        select, tuple_)
from sqlalchemy import inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.sql import insert as sa_insert, text, update

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "takken.db"
UPLOAD_DIR = DATA_DIR / "uploads"
REJECT_DIR = DATA_DIR / "rejects"
OFFLINE_EXPORT_DIR = DATA_DIR / "offline_exports"
MAPPING_KIND_QUESTIONS = "questions"
MAPPING_KIND_ANSWERS = "answers"
DEFAULT_CATEGORY_MAP = {
    "宅建業法": "宅建業法",
    "業法": "宅建業法",
    "権利関係": "権利関係",
    "民法": "権利関係",
    "法令上の制限": "法令上の制限",
    "制限": "法令上の制限",
    "税・その他": "税・その他",
    "税その他": "税・その他",
}
CATEGORY_CHOICES = ["宅建業法", "権利関係", "法令上の制限", "税・その他"]
DIFFICULTY_DEFAULT = 3
LAW_BASELINE_LABEL = "適用法令基準日（R6/4/1）"
LAW_REFERENCE_BASE_URL = "https://elaws.e-gov.go.jp/search?q={query}"

GLOBAL_SEARCH_SUGGESTIONS = [
    "重要事項説明",
    "抵当権",
    "都市計画法",
    "宅建業免許",
    "宅地建物取引士",
    "瑕疵担保",
]

FONT_SIZE_SCALE = {
    "やや小さい": 0.95,
    "標準": 1.0,
    "やや大きい": 1.1,
    "大きい": 1.2,
}

SUBJECT_PRESETS = {
    "バランスよく10問": {
        "categories": CATEGORY_CHOICES,
        "difficulty": (1, 5),
        "review_only": False,
        "topics": [],
    },
    "民法・権利関係を集中演習": {
        "categories": ["権利関係"],
        "difficulty": (2, 5),
        "review_only": False,
        "topics": [],
    },
    "弱点復習に集中": {
        "categories": CATEGORY_CHOICES,
        "difficulty": (1, 4),
        "review_only": True,
        "topics": [],
    },
}

metadata = MetaData()

questions_table = Table(
    "questions",
    metadata,
    Column("id", String, primary_key=True),
    Column("year", Integer, nullable=False),
    Column("q_no", Integer, nullable=False),
    Column("category", String, nullable=False),
    Column("topic", String),
    Column("question", String, nullable=False),
    Column("choice1", String, nullable=False),
    Column("choice2", String, nullable=False),
    Column("choice3", String, nullable=False),
    Column("choice4", String, nullable=False),
    Column("correct", Integer),
    Column("explanation", String),
    Column("difficulty", Integer, default=DIFFICULTY_DEFAULT),
    Column("tags", String),
    Column("dup_note", String),
    UniqueConstraint("year", "q_no", name="uq_questions_year_qno"),
)

attempts_table = Table(
    "attempts",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("question_id", String, nullable=False),
    Column("selected", Integer),
    Column("is_correct", Integer),
    Column("seconds", Integer),
    Column("mode", String),
    Column("exam_id", Integer),
    Column("confidence", Integer),
    Column("grade", Integer),
    Column("created_at", DateTime, server_default=func.now()),
)

exams_table = Table(
    "exams",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String),
    Column("started_at", DateTime),
    Column("finished_at", DateTime),
    Column("year_mode", String),
    Column("score", Integer),
)

srs_table = Table(
    "questions_srs",
    metadata,
    Column("question_id", String, primary_key=True),
    Column("repetition", Integer, default=0),
    Column("interval", Integer, default=1),
    Column("ease", Float, default=2.5),
    Column("due_date", Date),
    Column("last_grade", Integer),
    Column("updated_at", DateTime, server_default=func.now(), onupdate=func.now()),
)

import_logs_table = Table(
    "import_logs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("started_at", DateTime),
    Column("finished_at", DateTime),
    Column("files", Integer),
    Column("inserted", Integer),
    Column("updated", Integer),
    Column("rejected", Integer),
    Column("conflicts", Integer),
    Column("seconds", Float),
    Column("policy", String),
)

mapping_profiles_table = Table(
    "mapping_profiles",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String, nullable=False),
    Column("kind", String, nullable=False),
    Column("mapping_json", JSON, nullable=False),
    Column("created_at", DateTime, server_default=func.now()),
)

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile


def ensure_directories() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    REJECT_DIR.mkdir(exist_ok=True)
    OFFLINE_EXPORT_DIR.mkdir(exist_ok=True)


def inject_style(css: str, style_id: str) -> None:
    sanitized_css = css.strip()
    if not sanitized_css:
        return
    css_payload = json.dumps(sanitized_css)
    style_id_payload = json.dumps(style_id)
    script = Template(
        """
        <script>
        (function() {
            const css = $css;
            const styleId = $style_id;
            const doc = window.parent ? window.parent.document : document;
            let styleTag = doc.getElementById(styleId);
            if (!styleTag) {
                styleTag = doc.createElement('style');
                styleTag.id = styleId;
                doc.head.appendChild(styleTag);
            }
            styleTag.innerHTML = css;
        })();
        </script>
        """
    ).substitute(css=css_payload, style_id=style_id_payload)
    html(script, height=0)


def ensure_schema_migrations(engine: Engine) -> None:
    inspector = inspect(engine)
    with engine.begin() as conn:
        attempt_columns = {col["name"] for col in inspector.get_columns("attempts")}
        if "confidence" not in attempt_columns:
            conn.execute(text("ALTER TABLE attempts ADD COLUMN confidence INTEGER"))
        if "grade" not in attempt_columns:
            conn.execute(text("ALTER TABLE attempts ADD COLUMN grade INTEGER"))


def apply_user_preferences() -> None:
    settings = st.session_state.get("settings", {})
    theme = settings.get("theme", "ライト")
    font_label = settings.get("font_size", "標準")
    scale = FONT_SIZE_SCALE.get(font_label, 1.0)
    base_css = f"""
:root {{
    --takken-font-scale: {scale};
}}
[data-testid="stAppViewContainer"] * {{
    font-size: calc(1rem * var(--takken-font-scale));
}}
.takken-search-suggestions .stButton>button {{
    width: 100%;
    margin-bottom: 0.35rem;
}}
.takken-search-suggestions .stButton>button:hover {{
    border-color: #6366f1;
}}
"""
    if theme == "ダーク":
        theme_css = """
[data-testid="stAppViewContainer"] {
    background-color: #0e1117;
    color: #e7eefc;
}
[data-testid="stSidebar"] {
    background-color: #111827;
}
.stMetric, .stAlert {
    background-color: rgba(255, 255, 255, 0.04);
}
"""
    else:
        theme_css = """
[data-testid="stAppViewContainer"] {
    background-color: #f8fafc;
    color: #1f2933;
}
[data-testid="stSidebar"] {
    background-color: #ffffff;
}
"""
    inject_style(base_css + theme_css, "takken-theme-styles")


def build_search_dictionary(df: pd.DataFrame) -> List[str]:
    tokens: Set[str] = set(GLOBAL_SEARCH_SUGGESTIONS + CATEGORY_CHOICES)
    empty_series = pd.Series(dtype="object")
    tokens.update(str(cat).strip() for cat in df.get("category", empty_series).dropna())
    tokens.update(str(topic).strip() for topic in df.get("topic", empty_series).dropna())
    for tags in df.get("tags", empty_series).dropna():
        for token in re.split(r"[\s,;、/\\|]+", str(tags)):
            cleaned = token.strip()
            if len(cleaned) >= 2:
                tokens.add(cleaned)
    tokens = {token for token in tokens if token}
    return sorted(tokens)


def match_search_suggestions(dictionary: List[str], query: str, limit: int = 6) -> List[str]:
    if not dictionary:
        return []
    if not query:
        return dictionary[:limit]
    query_lower = query.lower()
    scored: List[Tuple[float, str]] = []
    for candidate in dictionary:
        candidate_lower = candidate.lower()
        score = fuzz.partial_ratio(query_lower, candidate_lower)
        if candidate_lower.startswith(query_lower):
            score += 20
        scored.append((score, candidate))
    scored.sort(key=lambda item: (-item[0], item[1]))
    filtered = [candidate for score, candidate in scored if score >= 30]
    if not filtered:
        filtered = [candidate for _, candidate in scored[:limit]]
    return filtered[:limit]


def trigger_global_search() -> None:
    query = str(st.session_state.get("global_search_input", "") or "").strip()
    st.session_state["global_search_query"] = query
    st.session_state["global_search_submitted"] = bool(query)


def clear_global_search() -> None:
    st.session_state["global_search_input"] = ""
    st.session_state["global_search_query"] = ""
    st.session_state["global_search_submitted"] = False
    st.session_state.pop("global_search_pending", None)


def request_clear_global_search() -> None:
    st.session_state["global_search_should_clear"] = True


def set_global_search_query(query: str) -> None:
    normalized = str(query or "").strip()
    st.session_state["global_search_pending"] = {
        "query": normalized,
        "submitted": bool(normalized),
    }
    st.session_state["global_search_query"] = normalized
    st.session_state["global_search_submitted"] = bool(normalized)


def safe_rerun() -> None:
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


QUESTION_TEMPLATE_COLUMNS = [
    "year",
    "q_no",
    "category",
    "topic",
    "question",
    "choice1",
    "choice2",
    "choice3",
    "choice4",
    "explanation",
    "difficulty",
    "tags",
]

ANSWER_TEMPLATE_COLUMNS = [
    "year",
    "q_no",
    "correct_number",
    "correct_label",
    "correct_text",
    "explanation",
    "difficulty",
    "tags",
]


@st.cache_data(show_spinner=False)
def get_template_archive() -> bytes:
    question_template = pd.DataFrame(
        [
            {
                "year": dt.datetime.now().year,
                "q_no": 1,
                "category": CATEGORY_CHOICES[0],
                "topic": "小分類の例",
                "question": "ここに問題文を入力してください。",
                "choice1": "選択肢1",
                "choice2": "選択肢2",
                "choice3": "選択肢3",
                "choice4": "選択肢4",
                "explanation": "解説を入力できます。",
                "difficulty": DIFFICULTY_DEFAULT,
                "tags": "タグ1;タグ2",
            }
        ],
        columns=QUESTION_TEMPLATE_COLUMNS,
    )
    answer_template = pd.DataFrame(
        [
            {
                "year": dt.datetime.now().year,
                "q_no": 1,
                "correct_number": 1,
                "correct_label": "A",
                "correct_text": "選択肢1",
                "explanation": "正答の解説を入力できます。",
                "difficulty": DIFFICULTY_DEFAULT,
                "tags": "タグ1;タグ2",
            }
        ],
        columns=ANSWER_TEMPLATE_COLUMNS,
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("questions_template.csv", question_template.to_csv(index=False))
        zf.writestr("answers_template.csv", answer_template.to_csv(index=False))
        q_excel = io.BytesIO()
        with pd.ExcelWriter(q_excel, engine="openpyxl") as writer:
            question_template.to_excel(writer, index=False, sheet_name="questions")
        zf.writestr("questions_template.xlsx", q_excel.getvalue())
        a_excel = io.BytesIO()
        with pd.ExcelWriter(a_excel, engine="openpyxl") as writer:
            answer_template.to_excel(writer, index=False, sheet_name="answers")
        zf.writestr("answers_template.xlsx", a_excel.getvalue())
        description = (
            "questions_template は設問データ、answers_template は正答データのサンプルです。\n"
            "不要な行は削除し、ご自身のデータを入力してからアップロードしてください。"
        )
        zf.writestr("README.txt", description)
    buffer.seek(0)
    return buffer.getvalue()


@st.cache_resource
def get_engine() -> Engine:
    ensure_directories()
    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
    metadata.create_all(engine)
    ensure_schema_migrations(engine)
    return engine


class DBManager:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def load_dataframe(self, table: Table) -> pd.DataFrame:
        with self.engine.connect() as conn:
            df = pd.read_sql(select(table), conn)
        return df

    def upsert_questions(self, df: pd.DataFrame) -> Tuple[int, int]:
        records = df.to_dict(orient="records")
        ids = [rec["id"] for rec in records if "id" in rec]
        year_qno_pairs = {
            (rec["year"], rec["q_no"]) for rec in records if "year" in rec and "q_no" in rec
        }
        inserted = 0
        updated = 0
        with self.engine.begin() as conn:
            if ids:
                existing_ids: Set[str] = set(
                    conn.execute(
                        select(questions_table.c.id).where(questions_table.c.id.in_(ids))
                    ).scalars()
                )
            else:
                existing_ids = set()

            if year_qno_pairs:
                existing_pairs = {
                    (row.year, row.q_no): row.id
                    for row in conn.execute(
                        select(
                            questions_table.c.year,
                            questions_table.c.q_no,
                            questions_table.c.id,
                        ).where(
                            tuple_(questions_table.c.year, questions_table.c.q_no).in_(
                                list(year_qno_pairs)
                            )
                        )
                    )
                }
            else:
                existing_pairs = {}

            for rec in records:
                rec_id = rec.get("id")
                year_qno = (rec.get("year"), rec.get("q_no"))
                update_values = {k: v for k, v in rec.items() if k != "id"}

                if rec_id in existing_ids:
                    conn.execute(
                        update(questions_table)
                        .where(questions_table.c.id == rec_id)
                        .values(**update_values)
                    )
                    updated += 1
                elif year_qno in existing_pairs:
                    existing_id = existing_pairs[year_qno]
                    conn.execute(
                        update(questions_table)
                        .where(questions_table.c.id == existing_id)
                        .values(**update_values)
                    )
                    updated += 1
                else:
                    conn.execute(sa_insert(questions_table).values(**rec))
                    inserted += 1
                    if rec_id:
                        existing_ids.add(rec_id)
                    if None not in year_qno:
                        existing_pairs[year_qno] = rec_id
        return inserted, updated

    def fetch_question(self, question_id: str) -> Optional[pd.Series]:
        with self.engine.connect() as conn:
            df = pd.read_sql(select(questions_table).where(questions_table.c.id == question_id), conn)
        if df.empty:
            return None
        return df.iloc[0]

    def record_attempt(
        self,
        question_id: str,
        selected: Optional[int],
        is_correct: bool,
        seconds: int,
        mode: str,
        exam_id: Optional[int] = None,
        confidence: Optional[int] = None,
        grade: Optional[int] = None,
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                sa_insert(attempts_table).values(
                    question_id=question_id,
                    selected=selected,
                    is_correct=int(is_correct),
                    seconds=seconds,
                    mode=mode,
                    exam_id=exam_id,
                    confidence=confidence,
                    grade=grade,
                )
            )

    def fetch_srs(self, question_id: str) -> Optional[pd.Series]:
        with self.engine.connect() as conn:
            df = pd.read_sql(
                select(srs_table).where(srs_table.c.question_id == question_id),
                conn,
            )
        if df.empty:
            return None
        return df.iloc[0]

    def log_exam_result(self, payload: Dict[str, object]) -> Optional[int]:
        with self.engine.begin() as conn:
            result = conn.execute(sa_insert(exams_table).values(**payload))
            inserted = result.inserted_primary_key
        if inserted:
            return inserted[0]
        return None

    def update_question_fields(
        self,
        question_id: str,
        fields: Dict[str, Optional[str]],
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(questions_table)
                .where(questions_table.c.id == question_id)
                .values(**fields)
            )

    def get_attempt_stats(self) -> pd.DataFrame:
        with self.engine.connect() as conn:
            df = pd.read_sql(
                text(
                    """
                    SELECT
                        a.question_id,
                        q.category,
                        q.topic,
                        q.year,
                        a.is_correct,
                        a.created_at,
                        a.seconds,
                        a.selected,
                        a.mode,
                        a.confidence,
                        a.grade
                    FROM attempts a
                    JOIN questions q ON q.id = a.question_id
                    """
                ),
                conn,
            )
        return df

    def get_due_srs(self) -> pd.DataFrame:
        today = dt.date.today()
        with self.engine.connect() as conn:
            df = pd.read_sql(
                select(srs_table, questions_table.c.question, questions_table.c.category)
                .where(srs_table.c.question_id == questions_table.c.id)
                .where((srs_table.c.due_date <= today) | (srs_table.c.due_date.is_(None))),
                conn,
            )
        return df

    def upsert_srs(self, question_id: str, payload: Dict[str, Optional[str]]) -> None:
        with self.engine.begin() as conn:
            stmt = sqlite_insert(srs_table).values(question_id=question_id, **payload)
            do_update = stmt.on_conflict_do_update(
                index_elements=[srs_table.c.question_id],
                set_={key: getattr(stmt.excluded, key) for key in payload},
            )
            conn.execute(do_update)

    def log_import(self, payload: Dict[str, Optional[str]]) -> None:
        with self.engine.begin() as conn:
            conn.execute(sa_insert(import_logs_table).values(**payload))

    def save_mapping_profile(self, name: str, kind: str, mapping: Dict[str, str]) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                sa_insert(mapping_profiles_table).values(
                    name=name,
                    kind=kind,
                    mapping_json=mapping,
                )
            )

    def fetch_mapping_profiles(self, kind: Optional[str] = None) -> pd.DataFrame:
        with self.engine.connect() as conn:
            stmt = select(mapping_profiles_table)
            if kind:
                stmt = stmt.where(mapping_profiles_table.c.kind == kind)
            df = pd.read_sql(stmt, conn)
        return df

    def initialize_from_csv(self) -> None:
        with self.engine.connect() as conn:
            existing = conn.execute(select(func.count()).select_from(questions_table)).scalar()
        if existing and existing > 0:
            return
        questions_path = DATA_DIR / "questions_sample.csv"
        answers_path = DATA_DIR / "answers_sample.csv"
        if not questions_path.exists() or not answers_path.exists():
            return
        df_q = pd.read_csv(questions_path)
        df_a = pd.read_csv(answers_path)
        df_q = normalize_questions(df_q)
        df_a = normalize_answers(df_a)
        merged, *_ = merge_questions_answers(df_q, df_a, policy={"explanation": "overwrite", "tags": "merge"})
        self.upsert_questions(merged)
        rebuild_tfidf_cache()


@st.cache_data(show_spinner=False)
def load_questions_df() -> pd.DataFrame:
    engine = get_engine()
    db = DBManager(engine)
    df = db.load_dataframe(questions_table)
    return df


@st.cache_resource
def get_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(stop_words=None, max_features=5000)


def rebuild_tfidf_cache() -> None:
    load_questions_df.clear()
    get_vectorizer.clear()
    load_questions_df()


def get_question_texts(df: pd.DataFrame) -> pd.Series:
    return df["question"].fillna("") + "\n" + df["explanation"].fillna("")


def compute_similarity(target_id: str, top_n: int = 3) -> pd.DataFrame:
    df = load_questions_df()
    if df.empty or target_id not in df["id"].values:
        return pd.DataFrame()
    texts = get_question_texts(df)
    vectorizer = get_vectorizer()
    matrix = vectorizer.fit_transform(texts)
    index = df.index[df["id"] == target_id][0]
    target_vec = matrix[index]
    sims = cosine_similarity(target_vec, matrix).flatten()
    df = df.assign(similarity=sims)
    df = df[df["id"] != target_id].nlargest(top_n, "similarity")
    return df[["id", "year", "q_no", "category", "question", "similarity"]]


def normalize_questions(df: pd.DataFrame, mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    df = df.copy()
    if mapping:
        df = df.rename(columns=mapping)
    required_cols = [
        "year",
        "q_no",
        "question",
        "choice1",
        "choice2",
        "choice3",
        "choice4",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"必要な列が不足しています: {col}")
    df["year"] = df["year"].astype(int)
    df["q_no"] = df["q_no"].astype(int)
    df["category"] = df.get("category", "").fillna("").apply(normalize_category)
    df["category"] = df["category"].replace("", CATEGORY_CHOICES[0])
    df["topic"] = df.get("topic", "").fillna("")
    df["explanation"] = df.get("explanation", "").fillna("")
    df["difficulty"] = (
        df.get("difficulty")
        .fillna(DIFFICULTY_DEFAULT)
        .replace("", DIFFICULTY_DEFAULT)
        .astype(int)
    )
    df["tags"] = df.get("tags", "").fillna("")
    if "id" not in df.columns or df["id"].isna().any():
        df["id"] = df.apply(generate_question_id, axis=1)
    df = df.drop_duplicates(subset=["id"])
    df = df[
        [
            "id",
            "year",
            "q_no",
            "category",
            "topic",
            "question",
            "choice1",
            "choice2",
            "choice3",
            "choice4",
            "explanation",
            "difficulty",
            "tags",
        ]
    ]
    return df


def normalize_answers(df: pd.DataFrame, mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    df = df.copy()
    if mapping:
        df = df.rename(columns=mapping)
    if "year" not in df.columns or "q_no" not in df.columns:
        raise ValueError("year と q_no は必須です")
    df["year"] = df["year"].astype(int)
    df["q_no"] = df["q_no"].astype(int)
    for col in ["explanation", "tags"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df


def validate_question_records(df: pd.DataFrame) -> List[str]:
    errors: List[str] = []
    required_cols = [
        "year",
        "q_no",
        "question",
        "choice1",
        "choice2",
        "choice3",
        "choice4",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        errors.append(f"必須列が不足しています: {', '.join(missing)}")
        return errors
    working = df.reset_index(drop=True)
    if "id" in working.columns:
        dup_ids = working[working["id"].notna() & working["id"].duplicated()]["id"].unique()
        if dup_ids.size > 0:
            errors.append(f"重複したIDが存在します: {', '.join(map(str, dup_ids[:5]))}")
    dup_keys = working.duplicated(subset=["year", "q_no"], keep=False)
    if dup_keys.any():
        duplicates = working.loc[dup_keys, ["year", "q_no"]].reset_index()
        sample = ", ".join(
            f"{row.year}年問{row.q_no} (行{row['index'] + 2})" for _, row in duplicates.head(5).iterrows()
        )
        errors.append(f"年度と問番の組み合わせが重複しています: {sample}")
    for row_number, row in enumerate(working.itertuples(index=False), start=2):
        year = getattr(row, "year", "?")
        q_no = getattr(row, "q_no", "?")
        label = f"{year}年問{q_no} (行{row_number})"
        question_text = str(getattr(row, "question", ""))
        if not question_text.strip():
            errors.append(f"{label}：問題文が空欄です。")
        choices = [str(getattr(row, f"choice{i}", "")).strip() for i in range(1, 5)]
        if any(choice == "" for choice in choices):
            errors.append(f"{label}：空欄の選択肢があります。")
        non_empty = [c for c in choices if c]
        if len(set(non_empty)) < len(non_empty):
            errors.append(f"{label}：選択肢が重複しています。")
    return errors


def validate_answer_records(df: pd.DataFrame) -> List[str]:
    errors: List[str] = []
    required_cols = ["year", "q_no", "correct_number"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        errors.append(f"必須列が不足しています: {', '.join(missing)}")
        return errors
    working = df.reset_index(drop=True)
    dup_keys = working.duplicated(subset=["year", "q_no"], keep=False)
    if dup_keys.any():
        duplicates = working.loc[dup_keys, ["year", "q_no"]].reset_index()
        sample = ", ".join(
            f"{row.year}年問{row.q_no} (行{row['index'] + 2})" for _, row in duplicates.head(5).iterrows()
        )
        errors.append(f"年度と問番の組み合わせが重複しています: {sample}")
    if working["correct_number"].isna().any():
        rows = (working["correct_number"].isna().to_numpy().nonzero()[0] + 2).tolist()
        rows_text = ", ".join(map(str, rows[:5]))
        errors.append(f"correct_number に空欄があります (行 {rows_text})。")
    try:
        invalid = pd.to_numeric(working["correct_number"], errors="coerce")
    except Exception:
        invalid = pd.Series([np.nan] * len(working))
    out_of_range = working[(invalid < 1) | (invalid > 4) | invalid.isna()]
    if not out_of_range.empty:
        sample_rows = ", ".join(
            f"{row.year}年問{row.q_no} (行{row['index'] + 2})"
            for _, row in out_of_range.reset_index().head(5).iterrows()
        )
        errors.append(f"correct_number は1〜4の範囲で指定してください: {sample_rows}")
    return errors


def build_answers_export(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        correct_number = row.get("correct")
        if pd.isna(correct_number):
            correct_number = None
            correct_label = ""
            correct_text = ""
        else:
            correct_number = int(correct_number)
            correct_label = ["A", "B", "C", "D"][correct_number - 1]
            correct_text = str(row.get(f"choice{correct_number}", ""))
        records.append(
            {
                "year": row.get("year"),
                "q_no": row.get("q_no"),
                "correct_number": correct_number,
                "correct_label": correct_label,
                "correct_text": correct_text,
                "explanation": row.get("explanation", ""),
                "difficulty": row.get("difficulty"),
                "tags": row.get("tags", ""),
            }
        )
    return pd.DataFrame(records, columns=ANSWER_TEMPLATE_COLUMNS)


def build_sample_questions_csv() -> str:
    sample_rows = [
        {
            "year": 2023,
            "q_no": 1,
            "category": "宅建業法",
            "topic": "免許",
            "question": "宅地建物取引業者の免許について正しいものはどれか。",
            "choice1": "免許権者は必ず国土交通大臣である。",
            "choice2": "法人が免許を受ける場合、専任取引士は不要である。",
            "choice3": "免許替えの際は旧免許の有効期間を引き継げる。",
            "choice4": "知事免許業者が二以上の都道府県に事務所を設けるときは大臣免許が必要である。",
            "explanation": "宅建業法上、二以上の都道府県に事務所を設ける場合は大臣免許が必要。",
            "difficulty": 3,
            "tags": "宅建業法;免許",
        },
        {
            "year": 2023,
            "q_no": 2,
            "category": "権利関係",
            "topic": "物権変動",
            "question": "不動産物権変動の対抗要件に関する記述として正しいものはどれか。",
            "choice1": "不動産の贈与は口頭でも第三者に対抗できる。",
            "choice2": "所有権移転登記を備えなければ第三者に対抗できない。",
            "choice3": "仮登記のままでも常に第三者に優先する。",
            "choice4": "地上権設定は登記簿の記載を要しない。",
            "explanation": "不動産物権変動の対抗要件は原則として登記である。",
            "difficulty": 2,
            "tags": "権利関係;物権変動",
        },
    ]
    buffer = io.StringIO()
    pd.DataFrame(sample_rows, columns=QUESTION_TEMPLATE_COLUMNS).to_csv(buffer, index=False)
    return buffer.getvalue()


def build_sample_answers_csv() -> str:
    sample_rows = [
        {
            "year": 2023,
            "q_no": 1,
            "correct_number": 4,
            "correct_label": "D",
            "correct_text": "知事免許業者が二以上の都道府県に事務所を設けるときは大臣免許が必要。",
            "explanation": "宅建業法の免許制度に基づき、複数都道府県で営業する場合は大臣免許が必要です。",
            "difficulty": 3,
            "tags": "宅建業法;免許",
        },
        {
            "year": 2023,
            "q_no": 2,
            "correct_number": 2,
            "correct_label": "B",
            "correct_text": "所有権移転登記を備えなければ第三者に対抗できない。",
            "explanation": "不動産物権変動の対抗要件は登記が原則です。",
            "difficulty": 2,
            "tags": "権利関係;物権変動",
        },
    ]
    buffer = io.StringIO()
    pd.DataFrame(sample_rows, columns=ANSWER_TEMPLATE_COLUMNS).to_csv(buffer, index=False)
    return buffer.getvalue()


def merge_questions_answers(
    questions: pd.DataFrame,
    answers: pd.DataFrame,
    policy: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = questions.copy()
    if "correct" not in merged.columns:
        merged["correct"] = np.nan
    rejects_q = []
    rejects_a = []

    answer_map = {}
    for _, row in answers.iterrows():
        key = (row["year"], row["q_no"])
        answer_map[key] = row

    def determine_correct(row: pd.Series, ans_row: Optional[pd.Series]) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        if ans_row is None:
            return None, None, None
        correct_number = ans_row.get("correct_number")
        if pd.notna(correct_number):
            try:
                correct_number = int(correct_number)
                if correct_number in [1, 2, 3, 4]:
                    return correct_number, ans_row.get("explanation"), ans_row.get("tags")
            except ValueError:
                pass
        label = ans_row.get("correct_label")
        if pd.notna(label):
            label = str(label).strip().upper()
            mapping = {"Ａ": 1, "A": 1, "Ｂ": 2, "B": 2, "Ｃ": 3, "C": 3, "Ｄ": 4, "D": 4}
            if label in mapping:
                return mapping[label], ans_row.get("explanation"), ans_row.get("tags")
        text_answer = ans_row.get("correct_text")
        if pd.notna(text_answer) and text_answer:
            choices = [row.get(f"choice{i}", "") for i in range(1, 5)]
            for idx, choice in enumerate(choices, start=1):
                if str(choice).strip() == str(text_answer).strip():
                    return idx, ans_row.get("explanation"), ans_row.get("tags")
            ratios = [fuzz.ratio(str(choice), str(text_answer)) for choice in choices]
            max_ratio = max(ratios)
            if max_ratio >= 90:
                idx = ratios.index(max_ratio) + 1
                return idx, ans_row.get("explanation"), ans_row.get("tags")
            else:
                rejects_a.append({**ans_row.to_dict(), "reason": "選択肢と一致せず"})
                return None, None, None
        rejects_a.append({**ans_row.to_dict(), "reason": "正答情報が不足"})
        return None, None, None

    conflicts = []
    for idx, row in merged.iterrows():
        key = (row["year"], row["q_no"])
        ans_row = answer_map.get(key)
        correct, exp_override, tags_new = determine_correct(row, ans_row)
        if correct is not None:
            if pd.notna(row.get("correct")) and row.get("correct") != correct:
                conflicts.append({
                    "id": row["id"],
                    "year": row["year"],
                    "q_no": row["q_no"],
                    "existing": row.get("correct"),
                    "incoming": correct,
                })
            merged.at[idx, "correct"] = correct
        if ans_row is not None:
            if policy.get("explanation", "overwrite") == "overwrite" and pd.notna(ans_row.get("explanation")):
                merged.at[idx, "explanation"] = ans_row.get("explanation")
            elif policy.get("explanation") == "append":
                merged.at[idx, "explanation"] = (str(row.get("explanation", "")) + "\n" + str(ans_row.get("explanation", ""))).strip()
            if tags_new:
                if policy.get("tags", "merge") == "merge" and row.get("tags"):
                    tags_combined = set(str(row.get("tags", "")).split(";")) | set(str(tags_new).split(";"))
                    merged.at[idx, "tags"] = ";".join(sorted({t.strip() for t in tags_combined if t.strip()}))
                else:
                    merged.at[idx, "tags"] = tags_new
            if pd.notna(ans_row.get("difficulty")):
                merged.at[idx, "difficulty"] = int(ans_row.get("difficulty"))
    merged["correct"] = merged["correct"].fillna(0).astype(int).replace(0, np.nan)
    rejects_q_df = pd.DataFrame(rejects_q)
    rejects_a_df = pd.DataFrame(rejects_a)
    conflicts_df = pd.DataFrame(conflicts)
    return merged, rejects_q_df, rejects_a_df, conflicts_df


def normalize_category(value: str) -> str:
    if not value:
        return CATEGORY_CHOICES[0]
    value = str(value).strip()
    if value in CATEGORY_CHOICES:
        return value
    for key, target in DEFAULT_CATEGORY_MAP.items():
        if key in value:
            return target
    scores = {cat: fuzz.partial_ratio(value, cat) for cat in CATEGORY_CHOICES}
    best_cat = max(scores, key=scores.get)
    if scores[best_cat] >= 70:
        return best_cat
    return CATEGORY_CHOICES[0]


def generate_question_id(row: pd.Series) -> str:
    base = f"{row['year']}|{row['q_no']}|{str(row['question'])[:80]}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


@dataclass
class ExamSession:
    id: Optional[int]
    name: str
    questions: List[str]
    started_at: dt.datetime
    year_mode: str
    mode: str


@dataclass
class QuestionNavigation:
    has_prev: bool = False
    has_next: bool = False
    on_prev: Optional[Callable[[], None]] = None
    on_next: Optional[Callable[[], None]] = None
    label: Optional[str] = None


def select_random_questions(df: pd.DataFrame, count: int) -> List[str]:
    if df.empty:
        return []
    return random.sample(list(df["id"]), min(count, len(df)))


def stratified_exam(df: pd.DataFrame) -> List[str]:
    quotas = {"宅建業法": 20, "権利関係": 14, "法令上の制限": 8, "税・その他": 8}
    selected = []
    remaining = df.copy()
    for category, quota in quotas.items():
        subset = remaining[remaining["category"] == category]
        chosen = select_random_questions(subset, quota)
        selected.extend(chosen)
        remaining = remaining[~remaining["id"].isin(chosen)]
    if len(selected) < 50:
        additional = select_random_questions(remaining, 50 - len(selected))
        selected.extend(additional)
    return selected


def sm2_update(row: Optional[pd.Series], grade: int, initial_ease: float = 2.5) -> Dict[str, object]:
    today = dt.date.today()
    if row is None:
        repetition = 0
        interval = 1
        ease = initial_ease
    else:
        repetition = row.get("repetition", 0) or 0
        interval = row.get("interval", 1) or 1
        ease = row.get("ease", 2.5) or 2.5
    schedule = [1, 3, 7, 21]
    if grade >= 3:
        if repetition < len(schedule):
            interval = schedule[repetition]
        else:
            interval = int(round(max(interval, schedule[-1]) * ease))
        repetition += 1
    else:
        repetition = 0
        interval = 1
    ease = ease + (0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02))
    ease = max(ease, 1.3)
    due_date = today + dt.timedelta(days=interval)
    return {
        "repetition": repetition,
        "interval": interval,
        "ease": ease,
        "due_date": due_date,
        "last_grade": grade,
        "updated_at": dt.datetime.now(),
    }


def inject_ui_styles() -> None:
    if st.session_state.get("_ui_styles_injected"):
        return
    inject_style(
        """
.takken-choice-button button {
    width: 100%;
    min-height: 56px;
    font-size: 1.05rem;
    border-radius: 0.8rem;
}
.takken-choice-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 0.75rem;
    margin-bottom: 0.5rem;
}
@media (max-width: 768px) {
    .takken-choice-grid {
        grid-template-columns: 1fr;
    }
}
.takken-inline-actions button {
    min-height: 48px;
}
""",
        "takken-ui-styles",
    )
    st.session_state["_ui_styles_injected"] = True


def confidence_to_grade(is_correct: bool, confidence: int) -> int:
    confidence = max(0, min(100, confidence))
    if is_correct:
        if confidence >= 90:
            return 5
        if confidence >= 70:
            return 4
        if confidence >= 50:
            return 3
        return 2
    if confidence >= 70:
        return 1
    return 0


def get_offline_attempts_df() -> pd.DataFrame:
    records = st.session_state.get("offline_attempts", [])
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def persist_offline_attempts(df: pd.DataFrame) -> None:
    if df.empty:
        return
    OFFLINE_EXPORT_DIR.mkdir(exist_ok=True)
    csv_path = OFFLINE_EXPORT_DIR / "attempts_latest.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    parquet_path = OFFLINE_EXPORT_DIR / "attempts_latest.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
        st.session_state["_offline_parquet_error"] = None
    except Exception as exc:
        st.session_state["_offline_parquet_error"] = str(exc)
        if parquet_path.exists():
            parquet_path.unlink()


def log_offline_attempt(entry: Dict[str, object]) -> None:
    attempts = st.session_state.setdefault("offline_attempts", [])
    attempts.append(entry)
    df = get_offline_attempts_df()
    persist_offline_attempts(df)


def render_offline_downloads(key_prefix: str) -> None:
    df = get_offline_attempts_df()
    if df.empty:
        return
    with st.expander("学習結果ダウンロード", expanded=False):
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            "CSVをダウンロード",
            data=csv_buffer.getvalue(),
            file_name="takken_learning_log.csv",
            mime="text/csv",
            key=f"{key_prefix}_csv",
        )
        parquet_buffer = io.BytesIO()
        parquet_error = st.session_state.get("_offline_parquet_error")
        if parquet_error:
            st.warning(f"Parquetの自動保存に失敗しました: {parquet_error}")
        try:
            parquet_buffer.seek(0)
            df.to_parquet(parquet_buffer, index=False)
            st.download_button(
                "Parquetをダウンロード",
                data=parquet_buffer.getvalue(),
                file_name="takken_learning_log.parquet",
                mime="application/octet-stream",
                key=f"{key_prefix}_parquet",
            )
        except Exception as exc:
            st.warning(f"Parquetのダウンロード生成に失敗しました: {exc}")
        st.caption(f"ファイルは {OFFLINE_EXPORT_DIR.as_posix()} にも自動保存されます。")


def build_snippet(text: str, keyword: str, width: int = 80) -> str:
    if not text:
        return ""
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    match = pattern.search(text)
    cleaned = re.sub(r"\s+", " ", text)
    if not match:
        return (cleaned[:width] + "…") if len(cleaned) > width else cleaned
    start = max(0, match.start() - width // 2)
    end = min(len(cleaned), match.end() + width // 2)
    snippet = cleaned[start:end]
    if start > 0:
        snippet = "…" + snippet
    if end < len(cleaned):
        snippet = snippet + "…"
    return snippet


def search_questions(df: pd.DataFrame, query: str) -> pd.DataFrame:
    keywords = [kw.strip() for kw in re.split(r"[\s　]+", query) if kw.strip()]
    if not keywords:
        return df.iloc[0:0]
    mask = pd.Series(True, index=df.index)
    choice_cols = [f"choice{i}" for i in range(1, 5)]
    for kw in keywords:
        pattern = re.escape(kw)
        col_mask = df["question"].fillna("").str.contains(pattern, case=False, na=False)
        col_mask |= df["topic"].fillna("").str.contains(pattern, case=False, na=False)
        col_mask |= df["tags"].fillna("").str.contains(pattern, case=False, na=False)
        for col in choice_cols:
            col_mask |= df[col].fillna("").str.contains(pattern, case=False, na=False)
        mask &= col_mask
    results = df[mask].copy()
    if results.empty:
        return results
    primary = keywords[0]
    results["snippet"] = results.apply(
        lambda row: build_snippet(
            "\n".join(
                [
                    str(row.get("question", "")),
                    *(str(row.get(f"choice{i}", "")) for i in range(1, 5)),
                    str(row.get("explanation", "")),
                ]
            ),
            primary,
        ),
        axis=1,
    )
    return results


def render_global_search_panel(db: DBManager, df: pd.DataFrame, query: str) -> None:
    st.markdown("## 🔍 横断検索結果")
    results = search_questions(df, query)
    if results.empty:
        st.info("該当する問題が見つかりませんでした。フィルタやキーワードを見直してください。")
        return
    with st.expander("並び替え・フィルタ", expanded=False):
        category_options = sorted({str(cat) for cat in results["category"].dropna()})
        selected_categories = st.multiselect(
            "分野で絞り込み",
            category_options,
            default=category_options,
            help="興味のある分野だけを表示します。",
            key="global_search_categories",
        )
        year_values = sorted({int(y) for y in results["year"].dropna().astype(int)})
        year_range: Optional[Tuple[int, int]]
        if year_values:
            min_year, max_year = year_values[0], year_values[-1]
            if min_year == max_year:
                st.caption(f"年度フィルタ: {min_year}年のみのデータです。")
                year_range = (min_year, max_year)
            else:
                year_range = st.slider(
                    "年度範囲",
                    min_year,
                    max_year,
                    (min_year, max_year),
                    help="学習したい年度に絞り込めます。",
                    key="global_search_year_range",
                )
        else:
            year_range = None
        snippet_keyword = st.text_input(
            "要約内フィルタ",
            key="global_search_snippet_filter",
            help="要約に含まれる語句でさらに絞り込みます。",
        )
        sort_option = st.selectbox(
            "並び順",
            ["関連度順", "年度 (新しい順)", "年度 (古い順)", "問番昇順"],
            help="検索結果の表示順序を変更できます。",
            key="global_search_sort",
        )
    filtered = results.copy()
    if selected_categories:
        filtered = filtered[filtered["category"].isin(selected_categories)]
    if year_range:
        filtered = filtered[filtered["year"].between(year_range[0], year_range[1])]
    if snippet_keyword:
        filtered = filtered[
            filtered["snippet"].str.contains(snippet_keyword, case=False, na=False)
            | filtered["question"].str.contains(snippet_keyword, case=False, na=False)
        ]
    if filtered.empty:
        st.info("条件に合致する結果がありません。フィルタを緩めて再検索してください。")
        return
    if sort_option == "年度 (新しい順)":
        filtered = filtered.sort_values(["year", "q_no"], ascending=[False, True])
    elif sort_option == "年度 (古い順)":
        filtered = filtered.sort_values(["year", "q_no"], ascending=[True, True])
    elif sort_option == "問番昇順":
        filtered = filtered.sort_values(["q_no", "year"], ascending=[True, False])
    else:
        filtered = filtered.sort_values(["year", "q_no"], ascending=[False, True])
    display = filtered.head(30)
    summary_df = display[
        ["year", "q_no", "category", "topic", "snippet", "id"]
    ].rename(
        columns={
            "year": "年度",
            "q_no": "問番",
            "category": "分野",
            "topic": "小分類",
            "snippet": "要約",
            "id": "問題ID",
        }
    )
    st.dataframe(
        summary_df.set_index("問題ID"),
        use_container_width=True,
        column_config={
            "年度": st.column_config.NumberColumn("年度", format="%d", help="クリックで並び替えできます。"),
            "問番": st.column_config.NumberColumn("問番", format="%d", help="問番号でソートできます。"),
            "分野": st.column_config.TextColumn("分野", help="カテゴリ名にマウスオーバーすると全文が表示されます。"),
            "小分類": st.column_config.TextColumn("小分類", help="細目分類での検索に役立ちます。"),
            "要約": st.column_config.TextColumn("要約", help="問題文や解説から自動生成したサマリーです。"),
        },
    )
    selected_id = st.selectbox(
        "検索結果から問題を開く",
        display["id"],
        format_func=lambda x: format_question_label(df, x),
        key="global_search_select",
    )
    row = df[df["id"] == selected_id].iloc[0]
    render_question_interaction(db, row, attempt_mode="search", key_prefix="search")


def get_review_candidate_ids(db: DBManager) -> Set[str]:
    review_ids: Set[str] = set()
    attempts = db.get_attempt_stats()
    if not attempts.empty:
        attempts["created_at"] = pd.to_datetime(attempts["created_at"])
        last_attempts = (
            attempts.sort_values("created_at").groupby("question_id", as_index=False).tail(1)
        )
        review_ids.update(last_attempts[last_attempts["is_correct"] == 0]["question_id"].tolist())
        low_conf = int(st.session_state["settings"].get("review_low_confidence_threshold", 60))
        if "confidence" in last_attempts.columns:
            confidence_series = last_attempts["confidence"].fillna(101)
            review_ids.update(
                last_attempts[confidence_series <= low_conf]["question_id"].tolist()
            )
        days_threshold = int(st.session_state["settings"].get("review_elapsed_days", 7))
        cutoff = dt.datetime.now() - dt.timedelta(days=days_threshold)
        review_ids.update(
            last_attempts[last_attempts["created_at"] <= cutoff]["question_id"].tolist()
        )
    srs_due = db.get_due_srs()
    if not srs_due.empty:
        review_ids.update(srs_due["question_id"].tolist())
    return {str(qid) for qid in review_ids if pd.notna(qid)}


def parse_explanation_sections(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    if not text:
        return "", []
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    sections: List[Tuple[str, str]] = []
    summary = ""
    for line in lines:
        match = re.match(r"^【([^】]+)】(.*)$", line)
        if match:
            label = match.group(1).strip()
            content = match.group(2).strip()
        else:
            label = "補足"
            content = line.strip()
        sections.append((label, content))
        if not summary and label in ("要点", "結論") and content:
            summary = content
    if not summary and lines:
        summary = lines[0]
    summary = summary.strip()
    if len(summary) > 80:
        summary = summary[:77] + "…"
    return summary, sections


def render_explanation_content(row: pd.Series) -> None:
    explanation = row.get("explanation", "")
    summary, sections = parse_explanation_sections(explanation)
    if not explanation:
        st.write("解説が未登録です。データ入出力から解答データを取り込みましょう。")
        return
    st.markdown(f"**要点版**：{summary}")
    with st.expander("詳細解説をひらく", expanded=False):
        for label, content in sections:
            if not content:
                continue
            if label == "ミニ図":
                st.markdown(f"**{label}**")
                st.markdown(content, unsafe_allow_html=True)
            else:
                st.markdown(f"- **{label}**：{content}")
        similar = compute_similarity(row["id"])
        if not similar.empty:
            st.markdown("#### 類似問題")
            st.dataframe(similar, use_container_width=True)


def estimate_theta(attempts: pd.DataFrame, df: pd.DataFrame) -> Optional[float]:
    if attempts.empty:
        return None
    merged = attempts.merge(
        df[["id", "difficulty"]], left_on="question_id", right_on="id", how="left"
    )
    merged = merged.dropna(subset=["difficulty"])
    if merged.empty:
        return None
    difficulties = (merged["difficulty"].astype(float) - 3.0) * 0.7
    responses = merged["is_correct"].astype(float)
    theta = 0.0
    for _ in range(10):
        logits = theta - difficulties
        probs = 1.0 / (1.0 + np.exp(-logits))
        gradient = np.sum(responses - probs)
        hessian = -np.sum(probs * (1 - probs))
        if abs(hessian) < 1e-6:
            break
        theta -= gradient / hessian
        if abs(gradient) < 1e-4:
            break
    return float(theta)


def recommend_adaptive_questions(
    df: pd.DataFrame,
    attempts: pd.DataFrame,
    theta: float,
    limit: int = 10,
    low_conf_threshold: int = 70,
) -> pd.DataFrame:
    candidates = df.copy()
    candidates["difficulty"] = candidates["difficulty"].fillna(DIFFICULTY_DEFAULT)
    candidates["difficulty_scaled"] = (candidates["difficulty"].astype(float) - 3.0) * 0.7
    candidates["priority"] = np.where(
        candidates["difficulty_scaled"] >= theta,
        candidates["difficulty_scaled"] - theta,
        (theta - candidates["difficulty_scaled"]) * 1.5,
    )
    if not attempts.empty:
        attempts["created_at"] = pd.to_datetime(attempts["created_at"])
        last_attempts = attempts.sort_values("created_at").groupby("question_id").tail(1)
        if "confidence" in last_attempts:
            confidence_series = last_attempts["confidence"].fillna(0)
        else:
            confidence_series = pd.Series(0, index=last_attempts.index)
        mastered_ids = last_attempts[
            (last_attempts["is_correct"] == 1)
            & (confidence_series >= low_conf_threshold)
        ]["question_id"].tolist()
        if mastered_ids:
            candidates = candidates[~candidates["id"].isin(mastered_ids)]
    ranked = candidates.sort_values(["priority", "difficulty"], ascending=[True, False])
    return ranked.head(limit)


def compute_tricky_vocab_heatmap(
    attempts: pd.DataFrame, df: pd.DataFrame, top_n: int = 12
) -> pd.DataFrame:
    wrong = attempts[attempts["is_correct"] == 0]
    if wrong.empty:
        return pd.DataFrame()
    merged = wrong.merge(
        df[["id", "question", "category", "tags"]],
        left_on="question_id",
        right_on="id",
        how="left",
    )
    records: List[Dict[str, object]] = []
    pattern = re.compile(r"[一-龠ぁ-んァ-ンA-Za-z0-9]{2,}")
    for _, row in merged.iterrows():
        text = f"{row.get('question', '')} {row.get('tags', '')}"
        words = {w for w in pattern.findall(str(text)) if len(w) >= 2}
        for word in list(words)[:20]:
            records.append({"word": word, "category": row.get("category", "未分類")})
    if not records:
        return pd.DataFrame()
    freq = pd.DataFrame(records)
    counts = freq.groupby(["word", "category"]).size().reset_index(name="count")
    totals = counts.groupby("word")["count"].sum().reset_index(name="total")
    top_words = totals.nlargest(top_n, "total")["word"]
    heatmap_df = counts[counts["word"].isin(top_words)]
    return heatmap_df


def compute_most_improved_topic(attempts: pd.DataFrame, df: pd.DataFrame) -> Optional[Dict[str, object]]:
    merged = attempts.merge(df[["id", "topic"]], left_on="question_id", right_on="id", how="left")
    merged = merged.dropna(subset=["topic"])
    if merged.empty:
        return None
    improvements: List[Dict[str, object]] = []
    for topic, group in merged.groupby("topic"):
        if len(group) < 4:
            continue
        group = group.sort_values("created_at")
        window = max(1, len(group) // 3)
        early = group.head(window)["is_correct"].mean()
        late = group.tail(window)["is_correct"].mean()
        improvements.append(
            {
                "topic": topic,
                "delta": late - early,
                "early": early,
                "late": late,
                "attempts": len(group),
            }
        )
    if not improvements:
        return None
    best = max(improvements, key=lambda x: x["delta"])
    if best["delta"] <= 0:
        return None
    return best


def register_keyboard_shortcuts(mapping: Dict[str, str]) -> None:
    if not mapping:
        return
    html(
        """
        <script>
        (function() {
            const mapping = %s;
            document.addEventListener('keydown', function(event) {
                const active = document.activeElement;
                if (active && ['input', 'textarea', 'select'].includes(active.tagName.toLowerCase())) {
                    return;
                }
                const key = event.key ? event.key.toLowerCase() : '';
                const label = mapping[key];
                if (!label) {
                    return;
                }
                const doc = window.parent ? window.parent.document : document;
                const buttons = doc.querySelectorAll('button');
                for (const btn of buttons) {
                    if (!btn.innerText) {
                        continue;
                    }
                    if (btn.innerText.trim().startsWith(label.trim())) {
                        event.preventDefault();
                        btn.click();
                        break;
                    }
                }
            }, true);
        })();
        </script>
        """ % json.dumps({k.lower(): v for k, v in mapping.items()}, ensure_ascii=False),
        height=0,
    )
def decode_uploaded_file(file: "UploadedFile") -> List[Tuple[str, pd.DataFrame]]:
    filename = file.name
    suffix = Path(filename).suffix.lower()
    dataframes = []
    bytes_data = file.getvalue()
    if suffix == ".zip":
        with zipfile.ZipFile(io.BytesIO(bytes_data)) as z:
            for inner in z.infolist():
                if inner.is_dir():
                    continue
                inner_suffix = Path(inner.filename).suffix.lower()
                with z.open(inner) as f:
                    df = read_tabular(f.read(), inner_suffix)
                    dataframes.append((inner.filename, df))
    else:
        df = read_tabular(bytes_data, suffix)
        dataframes.append((filename, df))
    return dataframes


def read_tabular(data: bytes, suffix: str) -> pd.DataFrame:
    encoding_options = ["utf-8", "cp932"]
    if suffix in [".csv", ".txt", ".tsv", ""]:
        for enc in encoding_options:
            try:
                return pd.read_csv(io.BytesIO(data), encoding=enc)
            except Exception:
                continue
        return pd.read_csv(io.BytesIO(data))
    elif suffix in [".xlsx", ".xlsm", ".xls"]:
        return pd.read_excel(io.BytesIO(data))
    else:
        raise ValueError("サポートされていないファイル形式です")


def guess_dataset_kind(df: pd.DataFrame) -> str:
    cols = set(df.columns.str.lower())
    if {"choice1", "choice2", "choice3", "choice4"}.issubset(cols):
        return MAPPING_KIND_QUESTIONS
    if "correct_number" in cols or "correct_label" in cols or "correct_text" in cols:
        return MAPPING_KIND_ANSWERS
    return MAPPING_KIND_QUESTIONS


def store_uploaded_file(file: "UploadedFile", timestamp: str) -> Path:
    target_dir = UPLOAD_DIR / timestamp
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / file.name
    with open(path, "wb") as f:
        f.write(file.getbuffer())
    return path


def init_session_state() -> None:
    defaults = {
        "nav": "ホーム",
        "current_question": None,
        "attempt_start": None,
        "exam_session": None,
        "import_state": {},
        "global_search_input": "",
        "global_search_query": "",
        "global_search_submitted": False,
        "global_search_should_clear": False,
        "global_search_pending": None,
        "settings": {
            "shuffle_choices": True,
            "theme": "ライト",
            "font_size": "標準",
            "timer": True,
            "sm2_initial_ease": 2.5,
            "auto_advance": False,
            "review_low_confidence_threshold": 60,
            "review_elapsed_days": 7,
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main() -> None:
    st.set_page_config(page_title="宅建10年ドリル", layout="wide")
    init_session_state()
    if st.session_state.get("global_search_should_clear"):
        clear_global_search()
        st.session_state["global_search_should_clear"] = False
    pending_search = st.session_state.pop("global_search_pending", None)
    if pending_search:
        query = str(pending_search.get("query", "") or "").strip()
        submitted = bool(pending_search.get("submitted"))
        st.session_state["global_search_input"] = query
        st.session_state["global_search_query"] = query
        st.session_state["global_search_submitted"] = submitted
    apply_user_preferences()
    engine = get_engine()
    db = DBManager(engine)
    db.initialize_from_csv()
    df = load_questions_df()
    search_dictionary = build_search_dictionary(df)

    sidebar = st.sidebar
    sidebar.title("宅建10年ドリル")
    sidebar.text_input(
        "🔍 横断検索",
        key="global_search_input",
        placeholder="抵当権 代価弁済 / 再建築不可 など",
        help="Enterキーまたは『検索』ボタンで実行します。",
        on_change=trigger_global_search,
    )
    suggestion_container = sidebar.container()
    current_input = st.session_state.get("global_search_input", "")
    suggestions = match_search_suggestions(search_dictionary, current_input)
    if suggestions:
        with suggestion_container:
            st.caption("候補キーワード (クリックで検索欄に反映されます)")
            suggestion_cols = st.columns(2)
            for idx, keyword in enumerate(suggestions):
                col = suggestion_cols[idx % 2]
                if col.button(keyword, key=f"global_suggest_{idx}", type="secondary"):
                    set_global_search_query(keyword)
                    safe_rerun()
    search_action_cols = sidebar.columns(2)
    if search_action_cols[0].button("検索", key="global_search_button"):
        trigger_global_search()
    if search_action_cols[1].button(
        "条件クリア",
        key="global_search_clear",
        type="secondary",
    ):
        request_clear_global_search()
        safe_rerun()
    search_query = st.session_state.get("global_search_query", "")
    with sidebar.expander("キーワードヒント", expanded=False):
        st.caption("よく使う語句をクリックすると検索欄に自動入力されます。")
        hint_cols = st.columns(2)
        for idx, keyword in enumerate(GLOBAL_SEARCH_SUGGESTIONS):
            if hint_cols[idx % 2].button(keyword, key=f"global_search_hint_{idx}"):
                set_global_search_query(keyword)
                search_query = keyword
                safe_rerun()
    sidebar.divider()
    nav = sidebar.radio(
        "メニュー",
        ["ホーム", "学習モード", "模試", "弱点復習", "統計", "データ入出力", "設定"],
        index=["ホーム", "学習モード", "模試", "弱点復習", "統計", "データ入出力", "設定"].index(st.session_state.get("nav", "ホーム")),
    )
    st.session_state["nav"] = nav
    with sidebar.expander("モード別の使い方ガイド", expanded=False):
        st.markdown(
            "\n".join(
                [
                    "- **ホーム**：進捗サマリーと最近のインポート履歴を確認できます。",
                    "- **学習モード**：目的別タブから本試験演習やドリル、適応学習を選択します。",
                    "- **模試**：年度や出題方式を指定して本番同様の模試を開始します。",
                    "- **弱点復習**：SRSの期限が来た問題をまとめて復習します。",
                    "- **統計**：分野別の成績や時間分析を把握できます。",
                    "- **データ入出力**：CSV/ZIPの取り込みや履歴エクスポートを行います。",
                    "- **設定**：タイマーやシャッフルなど学習体験の好みを調整します。",
                ]
            )
        )

    if search_query and st.session_state.get("global_search_submitted"):
        render_global_search_panel(db, df, search_query)
        st.divider()

    if nav == "ホーム":
        render_home(db, df)
    elif nav == "学習モード":
        render_learning(db, df)
    elif nav == "模試":
        render_mock_exam(db, df)
    elif nav == "弱点復習":
        render_srs(db)
    elif nav == "統計":
        render_stats(db, df)
    elif nav == "データ入出力":
        render_data_io(db)
    elif nav == "設定":
        render_settings()


def render_home(db: DBManager, df: pd.DataFrame) -> None:
    st.title("ホーム")
    attempts = db.get_attempt_stats()
    st.markdown("### サマリー")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("設問数", len(df))
    with col2:
        st.metric("学習履歴", len(attempts))
    with col3:
        coverage = attempts["year"].nunique() / max(df["year"].nunique(), 1) * 100 if not attempts.empty else 0
        st.metric("年度カバレッジ", f"{coverage:.0f}%")
    st.info("過去問データと解答データをアップロードして学習を開始しましょう。サイドバーの『データ入出力』から取り込めます。")
    st.markdown("### 最近のインポート")
    with db.engine.connect() as conn:
        logs = pd.read_sql(select(import_logs_table).order_by(import_logs_table.c.id.desc()).limit(5), conn)
    if logs.empty:
        st.write("インポート履歴がありません。")
    else:
        st.dataframe(logs)


def render_learning(db: DBManager, df: pd.DataFrame) -> None:
    st.title("学習モード")
    if df.empty:
        st.warning("設問データがありません。『データ入出力』からアップロードしてください。")
        return
    tabs = st.tabs(["本試験モード", "適応学習", "分野別ドリル", "年度別演習", "弱点克服モード"])
    with tabs[0]:
        render_full_exam_lane(db, df)
    with tabs[1]:
        render_adaptive_lane(db, df)
    with tabs[2]:
        render_subject_drill_lane(db, df)
    with tabs[3]:
        render_year_drill_lane(db, df)
    with tabs[4]:
        render_weakness_lane(db, df)


def render_full_exam_lane(db: DBManager, df: pd.DataFrame) -> None:
    st.subheader("本試験モード")
    st.caption("50問・120分の本試験同等環境で得点力と時間配分をチェックします。")
    if len(df) < 50:
        st.info("50問の出題には最低50問のデータが必要です。データを追加してください。")
        return
    session: Optional[ExamSession] = st.session_state.get("exam_session")
    if session is None or session.mode != "本試験モード":
        if st.button(
            "50問模試を開始",
            key="start_full_exam",
            help="本試験と同じ50問・120分構成で一気に演習します。結果は統計に反映されます。",
        ):
            questions = stratified_exam(df)
            if not questions:
                st.warning("出題可能な問題が不足しています。")
                return
            st.session_state.pop("exam_result_本試験モード", None)
            st.session_state["exam_session"] = ExamSession(
                id=None,
                name=f"本試験モード {dt.datetime.now():%Y%m%d-%H%M}",
                questions=questions,
                started_at=dt.datetime.now(),
                year_mode="層化ランダム50",
                mode="本試験モード",
            )
            session = st.session_state.get("exam_session")
    session = st.session_state.get("exam_session")
    if session and session.mode == "本試験モード":
        render_exam_session_body(db, df, session, key_prefix="main_exam")
    result = st.session_state.get("exam_result_本試験モード")
    if result:
        display_exam_result(result)


def render_adaptive_lane(db: DBManager, df: pd.DataFrame) -> None:
    st.subheader("適応学習")
    st.caption("回答履歴から能力θを推定し、伸びしろの大きい難度を優先出題します。")
    attempts = db.get_attempt_stats()
    if attempts.empty:
        st.info("学習履歴がまだありません。本試験モードやドリルで取り組んでみましょう。")
        return
    theta = estimate_theta(attempts, df)
    if theta is None:
        st.info("推定に必要な難易度データが不足しています。問題に難易度を設定してください。")
        return
    st.metric("推定能力θ", f"{theta:.2f}")
    low_conf = int(st.session_state["settings"].get("review_low_confidence_threshold", 60))
    recommended = recommend_adaptive_questions(df, attempts, theta, low_conf_threshold=low_conf)
    if recommended.empty:
        st.info("おすすめできる問題がありません。条件を見直すか、新しい問題を追加してください。")
        return
    st.markdown("#### 推奨問題リスト (上位10件)")
    display = recommended[["id", "year", "q_no", "category", "difficulty", "priority"]].rename(
        columns={
            "id": "問題ID",
            "year": "年度",
            "q_no": "問番",
            "category": "分野",
            "difficulty": "難易度",
            "priority": "推奨度",
        }
    )
    st.dataframe(display.set_index("問題ID"), use_container_width=True)
    selected_id = st.selectbox(
        "取り組む問題",
        recommended["id"],
        format_func=lambda x: format_question_label(df, x),
        key="adaptive_question_select",
    )
    row = df[df["id"] == selected_id].iloc[0]
    render_question_interaction(db, row, attempt_mode="adaptive", key_prefix="adaptive")


def render_subject_drill_lane(db: DBManager, df: pd.DataFrame) -> None:
    st.subheader("分野別ドリル")
    st.caption("民法・借地借家法・都市計画法・建築基準法・税・鑑定評価・宅建業法といったテーマをピンポイントで鍛えます。")
    with st.expander("出題条件", expanded=True):
        preset = st.selectbox(
            "クイックプリセット",
            list(SUBJECT_PRESETS.keys()),
            help="代表的な絞り込み条件をワンクリックで適用できます。",
            key="subject_preset",
        )
        if st.button("プリセットを適用", key="subject_apply_preset"):
            config = SUBJECT_PRESETS[preset]
            st.session_state["subject_categories"] = config["categories"]
            st.session_state["subject_difficulty"] = config["difficulty"]
            st.session_state["subject_review_only"] = config["review_only"]
            st.session_state["subject_topics"] = config.get("topics", [])
            st.session_state["subject_keyword"] = ""
            safe_rerun()
        categories = st.multiselect(
            "分野",
            CATEGORY_CHOICES,
            default=CATEGORY_CHOICES,
            key="subject_categories",
        )
        topic_options = sorted({t for t in df["topic"].dropna().unique() if str(t).strip()})
        selected_topics = st.multiselect(
            "テーマ",
            topic_options,
            default=[],
            key="subject_topics",
        )
        difficulties = st.slider(
            "難易度",
            1,
            5,
            (1, 5),
            key="subject_difficulty",
            help="1は易しい〜5は難しい問題です。",
        )
        keyword = st.text_input(
            "キーワードで絞り込み (問題文/タグ)",
            key="subject_keyword",
            help="語句を入力すると問題文とタグから部分一致で検索します。",
        )
        review_only = st.checkbox(
            "復習だけ表示 (誤答・低確信・経過日数)",
            value=st.session_state.get("subject_review_only", False),
            key="subject_review_only",
        )
    filtered = df[
        df["category"].isin(categories)
        & df["difficulty"].between(difficulties[0], difficulties[1])
    ]
    if selected_topics:
        filtered = filtered[filtered["topic"].isin(selected_topics)]
    if keyword:
        keyword_lower = keyword.lower()
        filtered = filtered[
            filtered["question"].str.lower().str.contains(keyword_lower)
            | filtered["tags"].fillna("").str.lower().str.contains(keyword_lower)
        ]
    if review_only:
        review_ids = get_review_candidate_ids(db)
        if not review_ids:
            st.info("復習対象の問題はありません。学習履歴を増やしてみましょう。")
            return
        filtered = filtered[filtered["id"].isin(review_ids)]
    if filtered.empty:
        st.warning("条件に合致する問題がありません。フィルタを調整してください。")
        return
    st.caption(f"現在の条件に合致する問題は {len(filtered)} 件です。")
    question_id = st.selectbox(
        "出題問題",
        filtered["id"],
        format_func=lambda x: format_question_label(filtered, x),
        key="subject_question_select",
    )
    row = filtered[filtered["id"] == question_id].iloc[0]
    render_question_interaction(db, row, attempt_mode="subject_drill", key_prefix="subject")


def render_year_drill_lane(db: DBManager, df: pd.DataFrame) -> None:
    st.subheader("年度別演習")
    st.caption("年度ごとの出題を通し演習し、本試験本番と同じ流れで知識を定着させます。")
    years = sorted(df["year"].unique(), reverse=True)
    if not years:
        st.info("年度情報が登録されていません。データを確認してください。")
        return
    selected_year = st.selectbox("年度", years, key="year_drill_year")
    subset = df[df["year"] == selected_year].sort_values("q_no")
    if subset.empty:
        st.warning("選択した年度の問題がありません。")
        return
    total = len(subset)
    progress_key = "year_drill_index"
    stored_year_key = "year_drill_current_year"
    if st.session_state.get(stored_year_key) != selected_year:
        st.session_state[stored_year_key] = selected_year
        st.session_state[progress_key] = 0
    index = st.session_state.get(progress_key, 0)
    index = max(0, min(index, total - 1))
    current_row = subset.iloc[index]
    st.progress((index + 1) / total)

    def go_prev() -> None:
        st.session_state[progress_key] = max(0, index - 1)

    def go_next() -> None:
        st.session_state[progress_key] = min(total - 1, index + 1)

    navigation = QuestionNavigation(
        has_prev=index > 0,
        has_next=index < total - 1,
        on_prev=go_prev,
        on_next=go_next,
        label=f"{index + 1}/{total} 問を学習中",
    )
    render_question_interaction(
        db,
        current_row,
        attempt_mode="year_drill",
        key_prefix="year",
        navigation=navigation,
    )


def render_weakness_lane(db: DBManager, df: pd.DataFrame) -> None:
    st.subheader("弱点克服モード")
    st.caption("誤答・低正答率・時間超過が目立つ問題を優先的に出題し、得点の底上げを図ります。")
    attempts = db.get_attempt_stats()
    if attempts.empty:
        st.info("学習履歴がまだありません。本試験モードやドリルで取り組んでみましょう。")
        return
    summary = (
        attempts.groupby(["question_id", "category"])
        .agg(
            attempts_count=("is_correct", "count"),
            correct_count=("is_correct", "sum"),
            avg_seconds=("seconds", "mean"),
        )
        .reset_index()
    )
    summary["accuracy"] = summary["correct_count"] / summary["attempts_count"].replace(0, np.nan)
    summary["accuracy"] = summary["accuracy"].fillna(0)
    summary["avg_seconds"] = summary["avg_seconds"].fillna(0)
    summary["priority"] = (1 - summary["accuracy"]) * summary["attempts_count"] + np.where(
        summary["avg_seconds"] > 90,
        1,
        0,
    )
    merged = summary.merge(df[["id", "year", "q_no", "question"]], left_on="question_id", right_on="id", how="left")
    merged = merged.sort_values(["priority", "accuracy"], ascending=[False, True])
    st.markdown("#### 優先出題リスト")
    with st.expander("並び替え・フィルタ", expanded=False):
        category_options = sorted({str(cat) for cat in merged["category"].dropna()})
        selected_categories = st.multiselect(
            "分野",
            category_options,
            default=category_options,
            help="重点的に復習したい分野を選びます。",
            key="weakness_categories",
        )
        max_attempts = int(merged["attempts_count"].max()) if not merged.empty else 1
        min_attempts = int(merged["attempts_count"].min()) if not merged.empty else 0
        if min_attempts == max_attempts:
            attempts_threshold = max_attempts
            st.caption(f"挑戦回数フィルタ: {max_attempts} 回のみのデータです。")
        else:
            attempts_threshold = st.slider(
                "最低挑戦回数",
                min_attempts,
                max_attempts,
                min(min_attempts + 1, max_attempts),
                help="指定回数以上取り組んだ問題を対象にします。",
                key="weakness_attempts_threshold",
            )
        accuracy_ceiling = st.slider(
            "正答率の上限 (%)",
            0,
            100,
            70,
            step=5,
            help="この値より正答率が高い問題はリストから除外します。",
            key="weakness_accuracy_ceiling",
        )
        sort_option = st.selectbox(
            "並び順",
            ["優先度が高い順", "正答率が低い順", "挑戦回数が多い順", "年度が新しい順"],
            help="復習リストの並び替え基準を変更します。",
            key="weakness_sort",
        )
    filtered = merged.copy()
    if selected_categories:
        filtered = filtered[filtered["category"].isin(selected_categories)]
    if attempts_threshold:
        filtered = filtered[filtered["attempts_count"] >= attempts_threshold]
    filtered = filtered[filtered["accuracy"] * 100 <= accuracy_ceiling]
    if sort_option == "正答率が低い順":
        filtered = filtered.sort_values(["accuracy", "attempts_count"], ascending=[True, False])
    elif sort_option == "挑戦回数が多い順":
        filtered = filtered.sort_values(["attempts_count", "accuracy"], ascending=[False, True])
    elif sort_option == "年度が新しい順":
        filtered = filtered.sort_values(["year", "priority"], ascending=[False, False])
    else:
        filtered = filtered.sort_values(["priority", "accuracy"], ascending=[False, True])
    if filtered.empty:
        st.info("条件に合致する弱点候補がありません。フィルタ設定を見直してください。")
        return
    display_df = filtered.head(15)[
        [
            "question_id",
            "category",
            "year",
            "q_no",
            "accuracy",
            "attempts_count",
            "avg_seconds",
        ]
    ].rename(
        columns={
            "question_id": "問題ID",
            "category": "分野",
            "year": "年度",
            "q_no": "問",
            "accuracy": "正答率",
            "attempts_count": "挑戦回数",
            "avg_seconds": "平均解答時間(秒)",
        }
    )
    display_df["正答率"] = display_df["正答率"].astype(float) * 100
    st.dataframe(
        display_df.set_index("問題ID"),
        use_container_width=True,
        column_config={
            "分野": st.column_config.TextColumn("分野", help="復習対象のカテゴリです。"),
            "年度": st.column_config.NumberColumn("年度", format="%d", help="最新年度をクリックでソートできます。"),
            "問": st.column_config.NumberColumn("問", format="%d", help="年度内での問題番号です。"),
            "正答率": st.column_config.NumberColumn("正答率", format="%.0f%%", help="低いほど優先的に復習したい問題です。"),
            "挑戦回数": st.column_config.NumberColumn("挑戦回数", format="%d", help="取り組んだ回数です。"),
            "平均解答時間(秒)": st.column_config.NumberColumn(
                "平均解答時間(秒)",
                format="%.1f",
                help="長考した問題はミスの温床になりがちです。",
            ),
        },
    )
    candidates = filtered[~filtered["id"].isna()]
    if candidates.empty:
        st.info("弱点候補の問題を特定できませんでした。履歴を増やしましょう。")
        return
    selected_qid = st.selectbox(
        "復習する問題",
        candidates["id"],
        format_func=lambda x: format_question_label(df, x),
        key="weakness_question",
    )
    row = df[df["id"] == selected_qid].iloc[0]
    render_question_interaction(db, row, attempt_mode="weakness", key_prefix="weakness")


def render_exam_session_body(
    db: DBManager,
    df: pd.DataFrame,
    session: ExamSession,
    key_prefix: str,
    pass_line: float = 0.7,
) -> None:
    st.subheader(session.name)
    if st.session_state["settings"].get("timer", True):
        elapsed = dt.datetime.now() - session.started_at
        remaining = max(0, 120 * 60 - int(elapsed.total_seconds()))
        minutes, seconds = divmod(remaining, 60)
        st.info(f"残り時間: {minutes:02d}:{seconds:02d}")
    responses: Dict[str, int] = {}
    choice_labels = ["①", "②", "③", "④"]
    for qid in session.questions:
        row_df = df[df["id"] == qid]
        if row_df.empty:
            continue
        row = row_df.iloc[0]
        st.markdown(f"### {row['year']}年 問{row['q_no']}")
        st.markdown(f"**{row['category']} / {row['topic']}**")
        render_law_reference(row)
        st.markdown(row["question"], unsafe_allow_html=True)
        options = [row.get(f"choice{i}", "") for i in range(1, 5)]
        option_map = {
            idx + 1: f"{choice_labels[idx]} {options[idx]}" if options[idx] else choice_labels[idx]
            for idx in range(4)
        }
        choice = st.radio(
            f"回答 ({qid})",
            list(option_map.keys()),
            format_func=lambda opt: option_map.get(opt, str(opt)),
            key=f"{key_prefix}_exam_{qid}",
            horizontal=True,
            index=None,
        )
        if choice is not None:
            responses[qid] = choice
    if st.button(
        "採点する",
        key=f"{key_prefix}_grade",
        help="現在の回答を保存し、正答率や分野別統計を表示します。",
    ):
        evaluate_exam_attempt(db, df, session, responses, pass_line)


def evaluate_exam_attempt(
    db: DBManager,
    df: pd.DataFrame,
    session: ExamSession,
    responses: Dict[str, int],
    pass_line: float,
) -> None:
    total_questions = len(session.questions)
    correct = 0
    per_category: Dict[str, Dict[str, int]] = {}
    wrong_choices: List[Dict[str, object]] = []
    attempt_records: List[Tuple[str, int, bool]] = []
    duration = max((dt.datetime.now() - session.started_at).total_seconds(), 1)
    avg_seconds = duration / max(len(responses), 1)
    for qid in session.questions:
        row_df = df[df["id"] == qid]
        if row_df.empty:
            continue
        row = row_df.iloc[0]
        correct_choice = int(row.get("correct") or 0)
        choice = responses.get(qid)
        is_correct = choice is not None and correct_choice == choice
        if is_correct:
            correct += 1
        category = row.get("category", "その他")
        stats = per_category.setdefault(category, {"total": 0, "correct": 0})
        stats["total"] += 1
        if is_correct:
            stats["correct"] += 1
        attempt_records.append((qid, choice, is_correct))
        if not is_correct and correct_choice in range(1, 5):
            wrong_choices.append(
                {
                    "question": f"{row['year']}年 問{row['q_no']}",
                    "selected": choice,
                    "correct": correct_choice,
                    "category": category,
                }
            )
    finished_at = dt.datetime.now()
    exam_id = db.log_exam_result(
        {
            "name": session.name,
            "started_at": session.started_at,
            "finished_at": finished_at,
            "year_mode": session.year_mode,
            "score": correct,
        }
    )
    for qid, choice, is_correct in attempt_records:
        db.record_attempt(
            qid,
            choice,
            is_correct,
            seconds=int(avg_seconds),
            mode=session.mode,
            exam_id=exam_id,
            confidence=None,
            grade=None,
        )
    accuracy = correct / max(total_questions, 1)
    remaining_time = max(0, 120 * 60 - int(duration))
    answered = len(responses)
    unanswered = total_questions - answered
    expected_final = correct + unanswered * (correct / max(answered, 1)) if answered else 0
    result_payload = {
        "score": correct,
        "total": total_questions,
        "accuracy": accuracy,
        "pass_line": pass_line,
        "per_category": per_category,
        "wrong_choices": wrong_choices,
        "remaining_time": remaining_time,
        "expected_final": expected_final,
        "mode": session.mode,
        "exam_id": exam_id,
    }
    st.session_state[f"exam_result_{session.mode}"] = result_payload
    st.session_state["exam_session"] = None


def display_exam_result(result: Dict[str, object]) -> None:
    score = result["score"]
    total = result["total"]
    accuracy = result["accuracy"]
    pass_line = result["pass_line"]
    status = "✅ 合格ライン到達" if accuracy >= pass_line else "⚠️ 合格ライン未達"
    st.markdown(f"### 採点結果 — {status}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("得点", f"{score} / {total}")
    with col2:
        st.metric("正答率", f"{accuracy * 100:.1f}%")
    with col3:
        threshold = int(total * pass_line)
        st.metric("合格ライン", f"{threshold} 点")
    st.progress(min(accuracy / max(pass_line, 1e-6), 1.0))
    remaining_minutes, remaining_seconds = divmod(int(result["remaining_time"]), 60)
    st.metric(
        "残り時間 × 想定到達点",
        f"{remaining_minutes:02d}:{remaining_seconds:02d} / {result['expected_final']:.1f} 点",
    )
    if result["per_category"]:
        radar_df = pd.DataFrame(
            [
                {
                    "category": category,
                    "accuracy": stats["correct"] / max(stats["total"], 1),
                }
                for category, stats in result["per_category"].items()
            ]
        )
        if not radar_df.empty:
            import altair as alt

            radar_df = pd.concat([radar_df, radar_df.iloc[[0]]], ignore_index=True)
            chart = (
                alt.Chart(radar_df)
                .mark_line(closed=True)
                .encode(
                    theta=alt.Theta("category", sort=None),
                    radius=alt.Radius("accuracy", scale=alt.Scale(domain=[0, 1])),
                )
                .properties(title="分野別スコアレーダー")
            )
            points = (
                alt.Chart(radar_df)
                .mark_point(size=80)
                .encode(
                    theta=alt.Theta("category", sort=None),
                    radius=alt.Radius("accuracy", scale=alt.Scale(domain=[0, 1])),
                )
            )
            st.altair_chart(chart + points, use_container_width=True)
    wrong_choices = result.get("wrong_choices", [])
    if wrong_choices:
        st.markdown("#### 誤答の代替正解肢傾向")
        wrong_df = pd.DataFrame(wrong_choices)
        option_map = {1: "①", 2: "②", 3: "③", 4: "④"}
        wrong_df["選択肢"] = wrong_df["selected"].map(option_map).fillna("未回答")
        wrong_df["正解肢"] = wrong_df["correct"].map({1: "①", 2: "②", 3: "③", 4: "④"})
        st.dataframe(
            wrong_df[["question", "category", "選択肢", "正解肢"]],
            use_container_width=True,
        )


def render_question_interaction(
    db: DBManager,
    row: pd.Series,
    attempt_mode: str,
    key_prefix: str,
    navigation: Optional[QuestionNavigation] = None,
) -> None:
    inject_ui_styles()
    last_question_key = f"{key_prefix}_last_question"
    feedback_key = f"{key_prefix}_feedback"
    selected_key = f"{key_prefix}_selected_{row['id']}"
    order_key = f"{key_prefix}_order_{row['id']}"
    explanation_key = f"{key_prefix}_explanation_{row['id']}"
    confidence_key = f"{key_prefix}_confidence_{row['id']}"
    help_state_key = f"{key_prefix}_help_visible"
    if st.session_state.get(last_question_key) != row["id"]:
        st.session_state[last_question_key] = row["id"]
        st.session_state.pop(feedback_key, None)
        st.session_state[selected_key] = None
        st.session_state[confidence_key] = 50
        st.session_state[order_key] = None
        st.session_state[explanation_key] = False
    choices = [row.get(f"choice{i}", "") for i in range(1, 5)]
    base_order = list(range(4))
    if st.session_state["settings"].get("shuffle_choices", True):
        random.seed(f"{row['id']}_{attempt_mode}")
        random.shuffle(base_order)
    if st.session_state.get(order_key) is None:
        st.session_state[order_key] = base_order.copy()
    order = st.session_state.get(order_key, base_order)
    choice_labels = ["①", "②", "③", "④"]
    st.markdown(f"### {row['year']}年 問{row['q_no']}")
    st.markdown(f"**{row['category']} / {row['topic']}**")
    render_law_reference(row)
    st.markdown(row["question"], unsafe_allow_html=True)
    selected_choice = st.session_state.get(selected_key)
    button_labels: List[str] = []
    for idx in range(0, len(order), 2):
        cols = st.columns(2)
        for col_idx in range(2):
            pos = idx + col_idx
            if pos >= len(order):
                continue
            actual_idx = order[pos]
            label_text = choices[actual_idx]
            display_label = f"{choice_labels[actual_idx]} {label_text}".strip()
            button_labels.append(display_label or choice_labels[actual_idx])
            button_key = f"{key_prefix}_choice_{row['id']}_{actual_idx}"
            button_type = "primary" if selected_choice == actual_idx else "secondary"
            with cols[col_idx]:
                st.markdown('<div class="takken-choice-button">', unsafe_allow_html=True)
                if st.button(
                    display_label or choice_labels[actual_idx],
                    key=button_key,
                    use_container_width=True,
                    type=button_type,
                ):
                    st.session_state[selected_key] = actual_idx
                    selected_choice = actual_idx
                st.markdown("</div>", unsafe_allow_html=True)
    st.caption("1〜4キーで選択肢を即答できます。E:解説 F:フラグ N/P:移動 H:ヘルプ R:SRSリセット")
    confidence_value = st.session_state.get(confidence_key)
    if confidence_value is None:
        confidence_value = 50
    else:
        confidence_value = int(confidence_value)
    confidence_value = st.slider(
        "確信度（ぜんぜん自信なし ↔ 完璧）",
        0,
        100,
        value=confidence_value,
        key=confidence_key,
    )
    show_explanation = st.session_state.get(explanation_key, False)
    flagged = row["id"] in set(st.session_state.get("review_flags", []))
    grade_label = "採点"
    explanation_label = "解説を隠す" if show_explanation else "解説を表示"
    flag_label = "フラグ解除" if flagged else "復習フラグ"
    help_label = "ヘルプ"
    auto_advance_enabled = st.session_state["settings"].get("auto_advance", False)
    action_cols = st.columns(5)
    with action_cols[0]:
        grade_clicked = st.button(
            grade_label,
            key=f"{key_prefix}_grade_{row['id']}",
            use_container_width=True,
            type="primary",
        )
    with action_cols[1]:
        if st.button(
            explanation_label,
            key=f"{key_prefix}_toggle_explanation_{row['id']}",
            use_container_width=True,
        ):
            show_explanation = not show_explanation
            st.session_state[explanation_key] = show_explanation
    with action_cols[2]:
        if st.button(
            flag_label,
            key=f"{key_prefix}_flag_{row['id']}",
            use_container_width=True,
        ):
            flags = set(st.session_state.get("review_flags", []))
            if flagged:
                flags.discard(row["id"])
            else:
                flags.add(row["id"])
            st.session_state["review_flags"] = list(flags)
            flagged = row["id"] in flags
    with action_cols[3]:
        help_visible = st.session_state.get(help_state_key, False)
        if st.button(
            help_label,
            key=f"{key_prefix}_help_{row['id']}",
            use_container_width=True,
        ):
            help_visible = not help_visible
            st.session_state[help_state_key] = help_visible
        else:
            help_visible = st.session_state.get(help_state_key, False)
    with action_cols[4]:
        if st.button(
            "SRSリセット",
            key=f"{key_prefix}_srs_reset_{row['id']}",
            use_container_width=True,
        ):
            db.upsert_srs(
                row["id"],
                {
                    "repetition": 0,
                    "interval": 1,
                    "ease": st.session_state["settings"].get("sm2_initial_ease", 2.5),
                    "due_date": dt.date.today(),
                    "last_grade": None,
                    "updated_at": dt.datetime.now(),
                },
            )
            st.success("SRSを初期化しました。明日から復習に再投入されます。")
    if auto_advance_enabled and navigation and navigation.has_next:
        st.caption("採点後0.8秒で次問に自動遷移します。")
    if flagged:
        st.caption("この問題は復習フラグが設定されています。")
    feedback = st.session_state.get(feedback_key)
    if grade_clicked:
        if selected_choice is None:
            st.warning("選択肢を選んでから採点してください。")
        else:
            correct_choice = row.get("correct")
            if pd.isna(correct_choice):
                st.warning("正答が未登録の問題です。解答データを取り込んでください。")
            else:
                correct_choice = int(correct_choice)
                is_correct = (selected_choice + 1) == correct_choice
                initial_ease = st.session_state["settings"].get("sm2_initial_ease", 2.5)
                srs_row = db.fetch_srs(row["id"])
                grade_value = confidence_to_grade(is_correct, confidence_value)
                payload = sm2_update(srs_row, grade=grade_value, initial_ease=initial_ease)
                db.upsert_srs(row["id"], payload)
                log_offline_attempt(
                    {
                        "timestamp": dt.datetime.now().isoformat(),
                        "question_id": row["id"],
                        "year": row.get("year"),
                        "q_no": row.get("q_no"),
                        "category": row.get("category"),
                        "topic": row.get("topic"),
                        "selected": selected_choice + 1,
                        "correct": correct_choice,
                        "is_correct": is_correct,
                        "mode": attempt_mode,
                        "confidence": confidence_value,
                        "srs_grade": grade_value,
                    }
                )
                st.session_state[feedback_key] = {
                    "is_correct": is_correct,
                    "correct_choice": correct_choice,
                    "question_id": row["id"],
                    "confidence": confidence_value,
                    "grade": grade_value,
                }
                feedback = st.session_state[feedback_key]
                db.record_attempt(
                    row["id"],
                    selected_choice + 1,
                    is_correct,
                    seconds=0,
                    mode=attempt_mode,
                    confidence=confidence_value,
                    grade=grade_value,
                )
                if (
                    auto_advance_enabled
                    and navigation is not None
                    and navigation.has_next
                    and navigation.on_next is not None
                ):
                    time.sleep(0.8)
                    navigation.on_next()
                    st.experimental_rerun()
    if feedback and feedback.get("question_id") == row["id"]:
        correct_msg = choice_labels[feedback["correct_choice"] - 1]
        message = "正解です！" if feedback["is_correct"] else f"不正解。正答は {correct_msg}"
        (st.success if feedback["is_correct"] else st.error)(message)
        st.caption(
            f"確信度 {feedback.get('confidence', confidence_value)}% → 復習グレード {feedback.get('grade', '')}"
        )
    if show_explanation:
        st.markdown("#### 解説")
        render_explanation_content(row)
    if help_visible:
        st.info(
            """ショートカット一覧\n- 1〜4: 選択肢を選ぶ\n- E: 解説の表示/非表示\n- F: 復習フラグの切り替え\n- N/P: 次へ・前へ\n- H: このヘルプ"""
        )
    nav_prev_label = "前へ"
    nav_next_label = "次へ"
    if navigation:
        nav_cols = st.columns([1, 1, 2])
        prev_kwargs = {
            "use_container_width": True,
            "disabled": not navigation.has_prev,
            "key": f"{key_prefix}_prev_{row['id']}",
        }
        next_kwargs = {
            "use_container_width": True,
            "disabled": not navigation.has_next,
            "key": f"{key_prefix}_next_{row['id']}",
        }
        if navigation.on_prev:
            prev_kwargs["on_click"] = navigation.on_prev
        if navigation.on_next:
            next_kwargs["on_click"] = navigation.on_next
        with nav_cols[0]:
            st.button(nav_prev_label, **prev_kwargs)
        with nav_cols[1]:
            st.button(nav_next_label, **next_kwargs)
        with nav_cols[2]:
            if navigation.label:
                st.caption(navigation.label)
    render_offline_downloads(f"{key_prefix}_{row['id']}")
    shortcut_map: Dict[str, str] = {}
    for idx, label in enumerate(button_labels[:4]):
        shortcut_map[str(idx + 1)] = label
    shortcut_map["e"] = explanation_label
    shortcut_map["f"] = flag_label
    shortcut_map["h"] = help_label
    shortcut_map["r"] = "SRSリセット"
    if navigation:
        shortcut_map["n"] = nav_next_label
        shortcut_map["p"] = nav_prev_label
    register_keyboard_shortcuts(shortcut_map)

def format_question_label(df: pd.DataFrame, question_id: str) -> str:
    row = df[df["id"] == question_id].iloc[0]
    return f"{row['year']}年 問{row['q_no']} ({row['category']})"


def render_law_reference(row: pd.Series) -> None:
    query_source = row.get("tags") or row.get("topic") or row.get("category")
    if query_source:
        query = quote_plus(str(query_source).split(";")[0])
        url = LAW_REFERENCE_BASE_URL.format(query=query)
        st.caption(f"{LAW_BASELINE_LABEL} ｜ [条文検索]({url})")
    else:
        st.caption(LAW_BASELINE_LABEL)


def render_mock_exam(db: DBManager, df: pd.DataFrame) -> None:
    st.title("模試")
    if df.empty:
        st.warning("設問データがありません。")
        return
    with st.form("mock_exam_form"):
        year_mode = st.selectbox(
            "出題方式",
            ["最新年度", "年度選択", "層化ランダム50"],
            help="最新年度の全問、任意年度のみ、または分野バランスを取った50問から選べます。",
        )
        if year_mode == "年度選択":
            selected_year = st.selectbox(
                "年度",
                sorted(df["year"].unique(), reverse=True),
                help="模試に使用する年度を選択します。",
            )
            subset = df[df["year"] == selected_year]
            questions = list(subset["id"])
        elif year_mode == "最新年度":
            latest_year = df["year"].max()
            subset = df[df["year"] == latest_year]
            questions = list(subset["id"])
        else:
            questions = stratified_exam(df)
        submit = st.form_submit_button("模試開始", help="選択した条件で模試を開始し、即座に試験画面へ移動します。")
    if submit:
        st.session_state.pop("exam_result_模試", None)
        st.session_state["exam_session"] = ExamSession(
            id=None,
            name=f"模試 {dt.datetime.now():%Y%m%d-%H%M}",
            questions=questions,
            started_at=dt.datetime.now(),
            year_mode=year_mode,
            mode="模試",
        )
    session: Optional[ExamSession] = st.session_state.get("exam_session")
    if session and session.mode == "模試":
        render_exam_session_body(db, df, session, key_prefix="mock")
    result = st.session_state.get("exam_result_模試")
    if result:
        display_exam_result(result)


def render_srs(db: DBManager) -> None:
    st.title("弱点復習")
    due_df = db.get_due_srs()
    if due_df.empty:
        st.info("今日復習すべき問題はありません。")
        return
    for _, row in due_df.iterrows():
        st.markdown(f"### {row['question'][:40]}...")
        st.write(f"分野: {row['category']} / 期限: {row['due_date']}")
        grade = st.slider(
            f"評価 ({row['question_id']})",
            0,
            5,
            3,
            help="5=完全に覚えた、0=全く覚えていない。評価に応じて次回復習日が変わります。",
        )
        if st.button(
            "評価を保存",
            key=f"srs_save_{row['question_id']}",
            help="SM-2アルゴリズムに基づき次回の出題タイミングを更新します。",
        ):
            initial_ease = st.session_state["settings"].get("sm2_initial_ease", 2.5)
            payload = sm2_update(row, grade, initial_ease=initial_ease)
            db.upsert_srs(row["question_id"], payload)
            st.success("SRSを更新しました")


def render_stats(db: DBManager, df: pd.DataFrame) -> None:
    st.title("分析ダッシュボード")
    attempts = db.get_attempt_stats()
    if attempts.empty:
        st.info("統計情報はまだありません。学習を開始しましょう。")
        return
    try:
        attempts["created_at"] = pd.to_datetime(attempts["created_at"])
        attempts["seconds"] = pd.to_numeric(attempts.get("seconds"), errors="coerce")
        attempts["confidence"] = pd.to_numeric(attempts.get("confidence"), errors="coerce")
    except Exception as exc:
        st.error(f"学習履歴の整形に失敗しました ({exc})")
        st.info("CSVを直接編集した場合は、日付や秒数の列が数値・日時形式になっているか確認してください。")
        return
    question_meta_cols = ["id", "question", "category", "topic", "tags", "difficulty"]
    merged = attempts.merge(
        df[question_meta_cols],
        left_on="question_id",
        right_on="id",
        how="left",
        suffixes=("", "_question"),
    )
    for col in ["category", "topic"]:
        alt_col = f"{col}_question"
        if alt_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[col].fillna(merged[alt_col])
            else:
                merged[col] = merged[alt_col]
            merged = merged.drop(columns=[alt_col])
    if merged.empty:
        st.warning("集計対象の設問が特定できませんでした。設問データが削除されていないか確認してください。")
        st.info("『データ入出力』でquestions.csvを再度取り込み、設問IDと学習履歴の対応を復元できます。")
        return
    accuracy_series = merged["is_correct"].dropna()
    seconds_series = merged["seconds"].dropna()
    confidence_series = merged["confidence"].dropna()
    accuracy = accuracy_series.mean() if not accuracy_series.empty else np.nan
    avg_seconds = seconds_series.mean() if not seconds_series.empty else np.nan
    avg_confidence = confidence_series.mean() if not confidence_series.empty else np.nan
    st.subheader("サマリー")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("挑戦回数", f"{len(merged)} 回")
    with col2:
        accuracy_text = f"{accuracy * 100:.1f}%" if not np.isnan(accuracy) else "--"
        st.metric("平均正答率", accuracy_text)
    with col3:
        st.metric("平均解答時間", f"{avg_seconds:.1f} 秒" if not np.isnan(avg_seconds) else "--")
    if not np.isnan(avg_confidence):
        st.caption(f"平均確信度: {avg_confidence:.1f}%")
    else:
        st.caption("平均確信度: -- (十分なデータがありません)")

    import altair as alt

    st.subheader("分野別分析")
    category_stats = (
        merged.groupby("category")
        .agg(
            accuracy=("is_correct", "mean"),
            avg_seconds=("seconds", "mean"),
            attempts_count=("is_correct", "count"),
        )
        .reset_index()
    )
    if category_stats.empty:
        st.info("分野情報の十分なデータがありません。questions.csv の category 列を確認してください。")
    else:
        try:
            accuracy_chart = (
                alt.Chart(category_stats)
                .mark_bar()
                .encode(
                    x=alt.X("category", title="分野"),
                    y=alt.Y("accuracy", title="正答率", axis=alt.Axis(format="%")),
                    tooltip=["category", alt.Tooltip("accuracy", format=".2%"), "attempts_count"],
                )
                .properties(height=320)
            )
            st.altair_chart(accuracy_chart, use_container_width=True)
        except Exception as exc:
            st.warning(f"分野別正答率グラフを表示できませんでした ({exc})")
            st.caption("十分なデータが集まると自動で表示されます。")
        try:
            time_chart = (
                alt.Chart(category_stats)
                .mark_line(point=True)
                .encode(
                    x=alt.X("category", title="分野"),
                    y=alt.Y("avg_seconds", title="平均解答時間 (秒)", scale=alt.Scale(zero=False)),
                    tooltip=["category", alt.Tooltip("avg_seconds", format=".1f"), "attempts_count"],
                )
            )
            st.altair_chart(time_chart, use_container_width=True)
        except Exception as exc:
            st.warning(f"分野別時間グラフを表示できませんでした ({exc})")
            st.caption("十分なデータが集まると自動で表示されます。")

    st.subheader("確信度と正答の相関")
    valid_conf = merged.dropna(subset=["confidence"])
    if valid_conf.empty:
        st.info("確信度データがまだ十分ではありません。学習時にスライダーで自己評価してみましょう。")
    else:
        corr = valid_conf["confidence"].corr(valid_conf["is_correct"])
        st.metric("相関係数", f"{corr:.2f}")
        try:
            scatter = (
                alt.Chart(valid_conf)
                .mark_circle(opacity=0.6)
                .encode(
                    x=alt.X("confidence", title="確信度 (%)"),
                    y=alt.Y("is_correct", title="正答 (1=正解)", scale=alt.Scale(domain=[-0.1, 1.1])),
                    color=alt.Color("category", legend=None),
                    tooltip=["category", "topic", "confidence", "is_correct", "seconds"],
                )
            )
            st.altair_chart(scatter, use_container_width=True)
        except Exception as exc:
            st.warning(f"相関散布図を表示できませんでした ({exc})")
            st.caption("十分なデータが集まると自動で表示されます。")

    st.subheader("ひっかけ語彙ヒートマップ")
    heatmap_df = compute_tricky_vocab_heatmap(merged, df)
    if heatmap_df.empty:
        st.info("誤答語彙の十分なデータがありません。")
    else:
        try:
            word_order = (
                heatmap_df.groupby("word")["count"].sum().sort_values(ascending=False).index.tolist()
            )
            heatmap = (
                alt.Chart(heatmap_df)
                .mark_rect()
                .encode(
                    x=alt.X("category", title="分野"),
                    y=alt.Y("word", title="語彙", sort=word_order),
                    color=alt.Color("count", title="誤答回数", scale=alt.Scale(scheme="reds")),
                    tooltip=["word", "category", "count"],
                )
            )
            st.altair_chart(heatmap, use_container_width=True)
        except Exception as exc:
            st.warning(f"語彙ヒートマップを表示できませんでした ({exc})")
            st.caption("十分なデータが集まると自動で表示されます。")

    st.subheader("最も改善した論点")
    improvement = compute_most_improved_topic(merged, df)
    if improvement:
        st.success(
            f"{improvement['topic']}：正答率が {(improvement['early'] * 100):.1f}% → {(improvement['late'] * 100):.1f}% (＋{improvement['delta'] * 100:.1f}ポイント)"
        )
    else:
        st.info("改善の傾向を示す論点はまだ検出されていません。継続して学習しましょう。")
def render_data_io(db: DBManager) -> None:
    st.title("データ入出力")
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    st.markdown("### テンプレートファイル")
    st.download_button(
        "テンプレートをダウンロード (ZIP)",
        data=get_template_archive(),
        file_name=f"takken_templates_{timestamp}.zip",
        mime="application/zip",
    )
    st.caption("設問・正答データのCSV/XLSXテンプレートが含まれます。必要に応じて編集してご利用ください。")
    sample_cols = st.columns(2)
    with sample_cols[0]:
        st.download_button(
            "サンプル questions.csv",
            data=build_sample_questions_csv(),
            file_name="sample_questions.csv",
            mime="text/csv",
            help="Excelで開いて値を上書きすれば、そのまま取り込みできます。",
        )
    with sample_cols[1]:
        st.download_button(
            "サンプル answers.csv",
            data=build_sample_answers_csv(),
            file_name="sample_answers.csv",
            mime="text/csv",
            help="正答番号や解説の記入例です。コピーしてご利用ください。",
        )
    st.caption("サンプルCSVはExcelに貼り付けて使えるよう列幅を調整済みです。コピー&ペーストで手早く登録できます。")
    st.markdown("### クイックインポート (questions.csv / answers.csv)")
    quick_cols = st.columns(2)
    with quick_cols[0]:
        quick_questions_file = st.file_uploader(
            "questions.csv をアップロード",
            type=["csv"],
            key="quick_questions_file",
        )
    with quick_cols[1]:
        quick_answers_file = st.file_uploader(
            "answers.csv をアップロード",
            type=["csv"],
            key="quick_answers_file",
        )
    if st.button("クイックインポート実行", key="quick_import_button"):
        quick_errors: List[str] = []
        questions_df: Optional[pd.DataFrame] = None
        answers_df: Optional[pd.DataFrame] = None
        if quick_questions_file is None and quick_answers_file is None:
            st.warning("questions.csv か answers.csv のいずれかを選択してください。")
        else:
            if quick_questions_file is not None:
                data = quick_questions_file.getvalue()
                try:
                    questions_df = pd.read_csv(io.BytesIO(data))
                except UnicodeDecodeError:
                    questions_df = pd.read_csv(io.BytesIO(data), encoding="cp932")
                quick_errors.extend(validate_question_records(questions_df))
            if quick_answers_file is not None:
                data = quick_answers_file.getvalue()
                try:
                    answers_df = pd.read_csv(io.BytesIO(data))
                except UnicodeDecodeError:
                    answers_df = pd.read_csv(io.BytesIO(data), encoding="cp932")
                quick_errors.extend(validate_answer_records(answers_df))
            if quick_errors:
                for err in quick_errors:
                    st.error(err)
                st.info("テンプレートの列構成と突合してください。『テンプレートをダウンロード』から最新のCSVサンプルを取得できます。")
            else:
                policy = {"explanation": "overwrite", "tags": "merge"}
                merged_df: Optional[pd.DataFrame] = None
                rejects_q = pd.DataFrame()
                rejects_a = pd.DataFrame()
                conflicts = pd.DataFrame()
                normalization_failed = False
                if questions_df is not None:
                    try:
                        normalized_q = normalize_questions(questions_df)
                    except Exception as exc:
                        st.error(f"questions.csv の整形に失敗しました: {exc}")
                        normalization_failed = True
                        normalized_q = None
                else:
                    normalized_q = None
                if answers_df is not None:
                    try:
                        normalized_a = normalize_answers(answers_df)
                    except Exception as exc:
                        st.error(f"answers.csv の整形に失敗しました: {exc}")
                        normalization_failed = True
                        normalized_a = None
                else:
                    normalized_a = None
                if normalization_failed:
                    st.warning("列名や値の形式を見直してから再度インポートしてください。")
                else:
                    if normalized_q is not None and normalized_a is not None:
                        merged_df, rejects_q, rejects_a, conflicts = merge_questions_answers(
                            normalized_q, normalized_a, policy=policy
                        )
                    elif normalized_q is not None:
                        merged_df = normalized_q
                    elif normalized_a is not None:
                        existing = load_questions_df()
                        if existing.empty:
                            st.error("設問データが存在しません。answers.csv を取り込む前に questions.csv を読み込んでください。")
                        else:
                            merged_df, rejects_q, rejects_a, conflicts = merge_questions_answers(
                                existing, normalized_a, policy=policy
                            )
                    if merged_df is not None:
                        inserted, updated = db.upsert_questions(merged_df)
                        rebuild_tfidf_cache()
                        st.success(f"クイックインポートが完了しました。追加 {inserted} 件 / 更新 {updated} 件")
                        if not rejects_q.empty or not rejects_a.empty:
                            st.warning(
                                f"取り込めなかったレコードがあります。questions: {len(rejects_q)} 件 / answers: {len(rejects_a)} 件"
                            )
                            with st.expander("取り込めなかった行の詳細", expanded=False):
                                if not rejects_q.empty:
                                    st.markdown("**questions.csv**")
                                    st.dataframe(rejects_q.head(20))
                                if not rejects_a.empty:
                                    st.markdown("**answers.csv**")
                                    st.dataframe(rejects_a.head(20))
                                st.caption("理由列を参考にCSVの該当行を修正してください。全件はrejects_*.csvでダウンロードできます。")
                        if not conflicts.empty:
                            st.info(f"正答の衝突が {len(conflicts)} 件あり、上書きしました。")
    st.markdown("### クイックエクスポート (questions.csv / answers.csv)")
    existing_questions = load_questions_df()
    if existing_questions.empty:
        st.info("エクスポート可能な設問データがありません。")
    else:
        question_cols = QUESTION_TEMPLATE_COLUMNS.copy()
        if "id" in existing_questions.columns and "id" not in question_cols:
            question_cols.append("id")
        q_export = existing_questions[question_cols]
        q_buffer = io.StringIO()
        q_export.to_csv(q_buffer, index=False)
        st.download_button(
            "questions.csv をダウンロード",
            q_buffer.getvalue(),
            file_name="questions.csv",
            mime="text/csv",
            key="export_questions_csv",
        )
        answers_export = build_answers_export(existing_questions)
        a_buffer = io.StringIO()
        answers_export.to_csv(a_buffer, index=False)
        st.download_button(
            "answers.csv をダウンロード",
            a_buffer.getvalue(),
            file_name="answers.csv",
            mime="text/csv",
            key="export_answers_csv",
        )
    st.markdown("### (1) ファイル選択")
    uploaded_files = st.file_uploader(
        "設問・解答ファイルを選択 (CSV/XLSX/ZIP)",
        type=["csv", "xlsx", "xls", "zip"],
        accept_multiple_files=True,
    )
    datasets = []
    if uploaded_files:
        for file in uploaded_files:
            try:
                store_uploaded_file(file, timestamp)
                for name, df in decode_uploaded_file(file):
                    kind = guess_dataset_kind(df)
                    datasets.append({"name": name, "data": df, "kind": kind})
            except Exception as e:
                st.error(f"{file.name}: 読み込みに失敗しました ({e})")
    if not datasets:
        st.info("ファイルをアップロードしてください。")
        return

    st.markdown("### (2) プレビュー & マッピング")
    mapping_profiles = db.fetch_mapping_profiles()
    profile_options = ["新規マッピング"] + (mapping_profiles["name"].tolist() if not mapping_profiles.empty else [])
    selected_profile = st.selectbox("マッピングテンプレート", profile_options)
    profile_mapping: Dict[str, Dict[str, str]] = {}
    if selected_profile != "新規マッピング" and not mapping_profiles.empty:
        profile_row = mapping_profiles[mapping_profiles["name"] == selected_profile].iloc[0]
        mapping_data = profile_row["mapping_json"]
        if isinstance(mapping_data, str):
            try:
                profile_mapping = json.loads(mapping_data)
            except json.JSONDecodeError:
                profile_mapping = {}
        else:
            profile_mapping = mapping_data

    normalized_question_frames = []
    normalized_answer_frames = []
    conflict_resolutions: List[Dict[str, object]] = []

    policy = {
        "explanation": st.selectbox("解説の取り扱い", ["overwrite", "append"], format_func=lambda x: "上書き" if x == "overwrite" else "追記"),
        "tags": st.selectbox("タグの取り扱い", ["merge", "overwrite"], format_func=lambda x: "結合" if x == "merge" else "上書き"),
    }

    for dataset in datasets:
        df = dataset["data"]
        st.subheader(dataset["name"])
        st.dataframe(df.head())
        kind = st.selectbox(
            f"種別 ({dataset['name']})",
            [MAPPING_KIND_QUESTIONS, MAPPING_KIND_ANSWERS],
            index=0 if dataset["kind"] == MAPPING_KIND_QUESTIONS else 1,
        )
        dataset["kind"] = kind
        columns = df.columns.tolist()
        lower_map = {col.lower(): col for col in columns}
        if kind == MAPPING_KIND_QUESTIONS:
            mapping_targets = {
                "year": "年度",
                "q_no": "問番",
                "category": "大分類",
                "topic": "小分類",
                "question": "問題文",
                "choice1": "選択肢1",
                "choice2": "選択肢2",
                "choice3": "選択肢3",
                "choice4": "選択肢4",
                "explanation": "解説",
                "difficulty": "難易度",
                "tags": "タグ",
                "id": "ID",
            }
        else:
            mapping_targets = {
                "year": "年度",
                "q_no": "問番",
                "correct_number": "正答番号",
                "correct_label": "正答ラベル",
                "correct_text": "正答テキスト",
                "explanation": "解説",
                "difficulty": "難易度",
                "tags": "タグ",
            }
        saved_mapping = profile_mapping.get(dataset["name"], {}) if profile_mapping else {}
        mapping = {}
        for key, label in mapping_targets.items():
            default_idx = -1
            if saved_mapping and key in saved_mapping and saved_mapping[key] in columns:
                default_idx = columns.index(saved_mapping[key])
            elif key in lower_map:
                default_idx = columns.index(lower_map[key])
            selected_col = st.selectbox(
                f"{label}",
                ["未設定"] + columns,
                index=default_idx + 1 if default_idx >= 0 else 0,
                key=f"map_{dataset['name']}_{key}",
            )
            if selected_col != "未設定":
                mapping[key] = selected_col
        try:
            if kind == MAPPING_KIND_QUESTIONS:
                normalized = normalize_questions(df, mapping=mapping)
                normalized_question_frames.append(normalized)
            else:
                normalized = normalize_answers(df, mapping=mapping)
                normalized_answer_frames.append(normalized)
            dataset["mapping"] = mapping
        except Exception as e:
            st.error(f"マッピングエラー: {e}")

    if st.checkbox("マッピングをテンプレート保存"):
        profile_name = st.text_input("テンプレート名")
        if profile_name and st.button("保存"):
            mapping_payload = {ds["name"]: ds.get("mapping", {}) for ds in datasets}
            db.save_mapping_profile(profile_name, "mixed", mapping_payload)
            st.success("マッピングテンプレートを保存しました")

    if not normalized_question_frames:
        st.warning("設問データがありません。")
        return

    merged_questions = pd.concat(normalized_question_frames).drop_duplicates(subset=["id"])
    merged_answers = pd.concat(normalized_answer_frames) if normalized_answer_frames else pd.DataFrame()

    if not merged_answers.empty:
        merged, rejects_q, rejects_a, conflicts = merge_questions_answers(merged_questions, merged_answers, policy)
    else:
        merged = merged_questions
        rejects_q = pd.DataFrame()
        rejects_a = pd.DataFrame()
        conflicts = pd.DataFrame()

    if not conflicts.empty:
        st.error("正答情報のコンフリクトがあります。解決方法を選択してください。")
        with st.form("conflict_resolution_form"):
            for _, conflict in conflicts.iterrows():
                st.write(f"{int(conflict['year'])}年 問{int(conflict['q_no'])}")
                action = st.selectbox(
                    f"処理方法 ({conflict['id']})",
                    ["既存を維持", "解答で上書き", "手動修正"],
                    key=f"conflict_action_{conflict['id']}",
                )
                manual_value = st.number_input(
                    f"手動正答番号 ({conflict['id']})",
                    min_value=1,
                    max_value=4,
                    value=int(conflict["existing"]) if pd.notna(conflict["existing"]) and int(conflict["existing"]) in [1, 2, 3, 4] else 1,
                    key=f"conflict_manual_{conflict['id']}",
                )
                conflict_resolutions.append(
                    {
                        "id": conflict["id"],
                        "action": action,
                        "manual": manual_value,
                        "incoming": conflict["incoming"],
                        "existing": conflict["existing"],
                    }
                )
            applied = st.form_submit_button("解決を適用")
        if not applied:
            st.stop()
        for resolution in conflict_resolutions:
            if resolution["action"] == "解答で上書き":
                merged.loc[merged["id"] == resolution["id"], "correct"] = resolution["incoming"]
            elif resolution["action"] == "手動修正":
                merged.loc[merged["id"] == resolution["id"], "correct"] = resolution["manual"]
        conflicts = pd.DataFrame()

    st.markdown("### (3) 正規化 & バリデーション結果")
    st.success(f"設問{len(merged)}件を取り込みます。")
    if not rejects_a.empty:
        buffer = io.StringIO()
        rejects_a.to_csv(buffer, index=False)
        st.download_button("rejects_answers.csv をダウンロード", buffer.getvalue(), file_name="rejects_answers.csv", mime="text/csv")
    if not rejects_q.empty:
        buffer = io.StringIO()
        rejects_q.to_csv(buffer, index=False)
        st.download_button("rejects_questions.csv をダウンロード", buffer.getvalue(), file_name="rejects_questions.csv", mime="text/csv")

    if st.button("(4) 統合 (UPSERT) 実行"):
        started = dt.datetime.now()
        progress = st.progress(0)
        inserted, updated = db.upsert_questions(merged)
        progress.progress(70)
        finished = dt.datetime.now()
        seconds = (finished - started).total_seconds()
        policy_payload = {**policy, "conflict_resolutions": conflict_resolutions}
        db.log_import(
            {
                "started_at": started,
                "finished_at": finished,
                "files": len(uploaded_files),
                "inserted": inserted,
                "updated": updated,
                "rejected": len(rejects_a) + len(rejects_q),
                "conflicts": len(conflicts),
                "seconds": seconds,
                "policy": json.dumps(policy_payload, ensure_ascii=False),
            }
        )
        rebuild_tfidf_cache()
        progress.progress(100)
        st.success("インポートが完了しました。TF-IDFを再構築しました。")

    st.markdown("### (5) 履歴エクスポート")
    with db.engine.connect() as conn:
        attempts_df = pd.read_sql(select(attempts_table), conn)
        exams_df = pd.read_sql(select(exams_table), conn)
    if not attempts_df.empty:
        buffer = io.StringIO()
        attempts_df.to_csv(buffer, index=False)
        st.download_button("attempts.csv をダウンロード", buffer.getvalue(), file_name="attempts.csv", mime="text/csv")
    if not exams_df.empty:
        buffer = io.StringIO()
        exams_df.to_csv(buffer, index=False)
        st.download_button("exams.csv をダウンロード", buffer.getvalue(), file_name="exams.csv", mime="text/csv")
    if DB_PATH.exists():
        st.download_button("SQLiteバックアップをダウンロード", DB_PATH.read_bytes(), file_name="takken.db")

    st.markdown("### (6) データ消去")
    with st.form("data_reset_form"):
        reset_attempts = st.checkbox("学習履歴 (attempts) を削除")
        reset_exams = st.checkbox("模試結果 (exams) を削除")
        reset_all = st.checkbox("全データを初期化 (設問含む)")
        confirmed = st.form_submit_button("削除を実行")
    if confirmed:
        with db.engine.begin() as conn:
            if reset_all:
                for table in [attempts_table, exams_table, srs_table, questions_table]:
                    conn.execute(table.delete())
            else:
                if reset_attempts:
                    conn.execute(attempts_table.delete())
                if reset_exams:
                    conn.execute(exams_table.delete())
        rebuild_tfidf_cache()
        st.success("選択したデータを削除しました。")

    st.markdown("### (7) テンプレートダウンロード")
    with open(DATA_DIR / "questions_sample.csv", "rb") as f:
        st.download_button("設問テンプレCSV", f, file_name="questions_template.csv")
    with open(DATA_DIR / "answers_sample.csv", "rb") as f:
        st.download_button("解答テンプレCSV", f, file_name="answers_template.csv")


def render_settings() -> None:
    st.title("設定")
    settings = st.session_state["settings"]
    st.info("学習体験を自分好みにカスタマイズできます。各項目の説明を参考に調整してください。")
    settings["theme"] = st.selectbox(
        "テーマ",
        ["ライト", "ダーク"],
        index=0 if settings.get("theme") == "ライト" else 1,
        help="画面の配色を切り替えます。暗い環境ではダークテーマがおすすめです。",
    )
    size_options = list(FONT_SIZE_SCALE.keys())
    default_size = settings.get("font_size", "標準")
    size_index = size_options.index(default_size) if default_size in size_options else size_options.index("標準")
    settings["font_size"] = st.selectbox(
        "フォントサイズ",
        size_options,
        index=size_index,
        help="文字サイズを調整して読みやすさを最適化します。『大きい』は夜間学習や高解像度モニタ向きです。",
    )
    settings["shuffle_choices"] = st.checkbox(
        "選択肢をシャッフル",
        value=settings.get("shuffle_choices", True),
        help="毎回選択肢の順番をランダムに入れ替えて、位置記憶に頼らない訓練を行います。",
    )
    settings["timer"] = st.checkbox(
        "タイマーを表示",
        value=settings.get("timer", True),
        help="回答画面に経過時間を表示して本番同様の時間感覚を養います。",
    )
    settings["sm2_initial_ease"] = st.slider(
        "SM-2初期ease",
        1.3,
        3.0,
        settings.get("sm2_initial_ease", 2.5),
        help="間隔反復アルゴリズムの初期難易度です。既定値2.5で迷ったらそのままにしましょう。",
    )
    settings["auto_advance"] = st.checkbox(
        "採点後に自動で次問へ進む (0.8秒遅延)",
        value=settings.get("auto_advance", False),
        help="正誤判定後に待機せず次の問題へ進みたい場合に有効化します。",
    )
    settings["review_low_confidence_threshold"] = st.slider(
        "低確信として扱う確信度 (%)",
        0,
        100,
        int(settings.get("review_low_confidence_threshold", 60)),
        help="自己評価の確信度がこの値未満なら復習対象に含めます。",
    )
    settings["review_elapsed_days"] = st.slider(
        "復習抽出の経過日数しきい値",
        1,
        30,
        int(settings.get("review_elapsed_days", 7)),
        help="最終挑戦からこの日数が経過した問題を復習候補に追加します。",
    )
    if st.button("TF-IDFを再学習", help="検索精度が気になるときに再計算します。データ更新後の再実行がおすすめです。"):
        rebuild_tfidf_cache()
        st.success("TF-IDFを再学習しました")


if __name__ == "__main__":
    main()
