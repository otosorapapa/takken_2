import datetime as dt
import hashlib
import io
import json
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple
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
                        select)
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
        inserted = 0
        updated = 0
        with self.engine.begin() as conn:
            if ids:
                existing = set(
                    conn.execute(
                        select(questions_table.c.id).where(questions_table.c.id.in_(ids))
                    ).scalars()
                )
            else:
                existing = set()
            for rec in records:
                stmt = sqlite_insert(questions_table).values(**rec)
                do_update_stmt = stmt.on_conflict_do_update(
                    index_elements=[questions_table.c.id],
                    set_={
                        col.name: getattr(stmt.excluded, col.name)
                        for col in questions_table.columns
                        if col.name not in ("id",)
                    },
                )
                conn.execute(do_update_stmt)
                if rec["id"] in existing:
                    updated += 1
                else:
                    inserted += 1
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
                        q.year,
                        a.is_correct,
                        a.created_at,
                        a.seconds,
                        a.selected,
                        a.mode
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
    if grade >= 3:
        if repetition == 0:
            interval = 1
        elif repetition == 1:
            interval = 6
        else:
            interval = int(round(interval * ease))
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
    st.markdown(
        """
        <style>
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
        </style>
        """,
        unsafe_allow_html=True,
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
        "settings": {
            "shuffle_choices": True,
            "theme": "ライト",
            "timer": True,
            "sm2_initial_ease": 2.5,
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main() -> None:
    st.set_page_config(page_title="宅建10年ドリル", layout="wide")
    init_session_state()
    engine = get_engine()
    db = DBManager(engine)
    db.initialize_from_csv()

    st.sidebar.title("宅建10年ドリル")
    nav = st.sidebar.radio(
        "メニュー",
        ["ホーム", "学習モード", "模試", "弱点復習", "統計", "データ入出力", "設定"],
        index=["ホーム", "学習モード", "模試", "弱点復習", "統計", "データ入出力", "設定"].index(st.session_state.get("nav", "ホーム")),
    )
    st.session_state["nav"] = nav

    if nav == "ホーム":
        render_home(db)
    elif nav == "学習モード":
        render_learning(db)
    elif nav == "模試":
        render_mock_exam(db)
    elif nav == "弱点復習":
        render_srs(db)
    elif nav == "統計":
        render_stats(db)
    elif nav == "データ入出力":
        render_data_io(db)
    elif nav == "設定":
        render_settings()


def render_home(db: DBManager) -> None:
    st.title("ホーム")
    df = load_questions_df()
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


def render_learning(db: DBManager) -> None:
    st.title("学習モード")
    df = load_questions_df()
    if df.empty:
        st.warning("設問データがありません。『データ入出力』からアップロードしてください。")
        return
    tabs = st.tabs(["本試験モード", "分野別ドリル", "年度別演習", "弱点克服モード"])
    with tabs[0]:
        render_full_exam_lane(db, df)
    with tabs[1]:
        render_subject_drill_lane(db, df)
    with tabs[2]:
        render_year_drill_lane(db, df)
    with tabs[3]:
        render_weakness_lane(db, df)


def render_full_exam_lane(db: DBManager, df: pd.DataFrame) -> None:
    st.subheader("本試験モード")
    st.caption("50問・120分の本試験同等環境で得点力と時間配分をチェックします。")
    if len(df) < 50:
        st.info("50問の出題には最低50問のデータが必要です。データを追加してください。")
        return
    session: Optional[ExamSession] = st.session_state.get("exam_session")
    if session is None or session.mode != "本試験モード":
        if st.button("50問模試を開始", key="start_full_exam"):
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


def render_subject_drill_lane(db: DBManager, df: pd.DataFrame) -> None:
    st.subheader("分野別ドリル")
    st.caption("民法・借地借家法・都市計画法・建築基準法・税・鑑定評価・宅建業法といったテーマをピンポイントで鍛えます。")
    with st.expander("出題条件", expanded=True):
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
        difficulties = st.slider("難易度", 1, 5, (1, 5), key="subject_difficulty")
        keyword = st.text_input("キーワードで絞り込み (問題文/タグ)", key="subject_keyword")
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
    if filtered.empty:
        st.warning("条件に合致する問題がありません。フィルタを調整してください。")
        return
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
    st.dataframe(
        merged.head(10)[
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
        ),
        use_container_width=True,
    )
    candidates = merged[~merged["id"].isna()]
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
    if st.button("採点する", key=f"{key_prefix}_grade"):
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
    st.caption("1〜4キーで選択肢を即答できます。E:解説 F:フラグ N/P:移動 H:ヘルプ")
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
    action_cols = st.columns(4)
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
                db.record_attempt(
                    row["id"],
                    selected_choice + 1,
                    is_correct,
                    seconds=0,
                    mode=attempt_mode,
                )
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
    if feedback and feedback.get("question_id") == row["id"]:
        correct_msg = choice_labels[feedback["correct_choice"] - 1]
        message = "正解です！" if feedback["is_correct"] else f"不正解。正答は {correct_msg}"
        (st.success if feedback["is_correct"] else st.error)(message)
        st.caption(
            f"確信度 {feedback.get('confidence', confidence_value)}% → 復習グレード {feedback.get('grade', '')}"
        )
    if show_explanation:
        st.markdown("#### 解説")
        st.write(row.get("explanation", "解説が未登録です。"))
        similar = compute_similarity(row["id"])
        if not similar.empty:
            st.markdown("#### 類似問題")
            st.dataframe(similar)
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


def render_mock_exam(db: DBManager) -> None:
    st.title("模試")
    df = load_questions_df()
    if df.empty:
        st.warning("設問データがありません。")
        return
    with st.form("mock_exam_form"):
        year_mode = st.selectbox("出題方式", ["最新年度", "年度選択", "層化ランダム50"])
        if year_mode == "年度選択":
            selected_year = st.selectbox("年度", sorted(df["year"].unique(), reverse=True))
            subset = df[df["year"] == selected_year]
            questions = list(subset["id"])
        elif year_mode == "最新年度":
            latest_year = df["year"].max()
            subset = df[df["year"] == latest_year]
            questions = list(subset["id"])
        else:
            questions = stratified_exam(df)
        submit = st.form_submit_button("模試開始")
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
        grade = st.slider(f"評価 ({row['question_id']})", 0, 5, 3)
        if st.button("評価を保存", key=f"srs_save_{row['question_id']}"):
            initial_ease = st.session_state["settings"].get("sm2_initial_ease", 2.5)
            payload = sm2_update(row, grade, initial_ease=initial_ease)
            db.upsert_srs(row["question_id"], payload)
            st.success("SRSを更新しました")


def render_stats(db: DBManager) -> None:
    st.title("統計")
    attempts = db.get_attempt_stats()
    if attempts.empty:
        st.info("統計情報はまだありません。学習を開始しましょう。")
        return
    attempts["date"] = pd.to_datetime(attempts["created_at"]).dt.date
    attempts_group = attempts.groupby("date")["is_correct"].mean().reset_index()
    chart = altair_chart(attempts_group, "date", "is_correct", "日次正答率")
    st.altair_chart(chart, use_container_width=True)
    cat_group = attempts.groupby("category")["is_correct"].mean().reset_index()
    chart2 = altair_chart(cat_group, "category", "is_correct", "分野別正答率", mark="bar")
    st.altair_chart(chart2, use_container_width=True)


def altair_chart(df: pd.DataFrame, x: str, y: str, title: str, mark: str = "line"):
    import altair as alt

    chart = getattr(alt.Chart(df), mark)().encode(x=x, y=y).properties(title=title)
    return chart


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
    settings["theme"] = st.selectbox("テーマ", ["ライト", "ダーク"], index=0 if settings.get("theme") == "ライト" else 1)
    settings["shuffle_choices"] = st.checkbox("選択肢をシャッフル", value=settings.get("shuffle_choices", True))
    settings["timer"] = st.checkbox("タイマーを表示", value=settings.get("timer", True))
    settings["sm2_initial_ease"] = st.slider("SM-2初期ease", 1.3, 3.0, settings.get("sm2_initial_ease", 2.5))
    if st.button("TF-IDFを再学習"):
        rebuild_tfidf_cache()
        st.success("TF-IDFを再学習しました")


if __name__ == "__main__":
    main()
