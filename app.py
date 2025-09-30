import datetime as dt
import hashlib
import io
import json
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
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
                    SELECT q.category, q.year, a.is_correct, a.created_at, a.seconds
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


def sm2_update(row: Optional[pd.Series], grade: int) -> Dict[str, object]:
    today = dt.date.today()
    if row is None:
        repetition = 0
        interval = 1
        ease = 2.5
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
    with st.expander("フィルタ"):
        years = st.multiselect("年度", sorted(df["year"].unique()), default=sorted(df["year"].unique()))
        categories = st.multiselect("分野", CATEGORY_CHOICES, default=CATEGORY_CHOICES)
        difficulties = st.slider("難易度", 1, 5, (1, 5))
    filtered = df[
        df["year"].isin(years)
        & df["category"].isin(categories)
        & df["difficulty"].between(difficulties[0], difficulties[1])
    ]
    if filtered.empty:
        st.warning("該当する設問がありません。条件を調整してください。")
        return
    question_id = st.selectbox("出題", filtered["id"], format_func=lambda x: format_question_label(filtered, x))
    row = filtered[filtered["id"] == question_id].iloc[0]
    st.subheader(f"{row['year']}年 問{row['q_no']}")
    st.markdown(f"**{row['category']} / {row['topic']}**")
    choices = [row[f"choice{i}"] for i in range(1, 5)]
    if st.session_state["settings"].get("shuffle_choices", True):
        random.seed(question_id)
        order = list(range(4))
        random.shuffle(order)
    else:
        order = list(range(4))
    choice_labels = ["①", "②", "③", "④"]
    selected = st.radio(
        "解答を選択",
        order,
        format_func=lambda idx: f"{choice_labels[idx]} {choices[idx]}",
        key=f"answer_{question_id}",
    )
    if st.button("採点", key=f"grade_{question_id}"):
        correct_choice = row.get("correct")
        if pd.isna(correct_choice):
            st.warning("正答が未登録の問題です。解答データを取り込んでください。")
        else:
            correct_choice = int(correct_choice)
            is_correct = (selected + 1) == correct_choice
            db.record_attempt(question_id, selected + 1, is_correct, seconds=0, mode="learning")
            st.success("正解です！" if is_correct else f"不正解。正答は {choice_labels[correct_choice - 1]}")
        st.markdown("#### 解説")
        st.write(row.get("explanation", "解説が未登録です。"))
        similar = compute_similarity(question_id)
        if not similar.empty:
            st.markdown("#### 類似問題")
            st.dataframe(similar)
        if st.button("要復習に追加", key=f"srs_{question_id}"):
            srs_row = fetch_srs_row(db, question_id)
            payload = sm2_update(srs_row, grade=2)
            db.upsert_srs(question_id, payload)
            st.toast("SRSに追加しました", icon="✅")


def format_question_label(df: pd.DataFrame, question_id: str) -> str:
    row = df[df["id"] == question_id].iloc[0]
    return f"{row['year']}年 問{row['q_no']} ({row['category']})"


def fetch_srs_row(db: DBManager, question_id: str) -> Optional[pd.Series]:
    with db.engine.connect() as conn:
        df = pd.read_sql(
            select(srs_table).where(srs_table.c.question_id == question_id),
            conn,
        )
    if df.empty:
        return None
    return df.iloc[0]


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
        st.session_state["exam_session"] = ExamSession(
            id=None,
            name=f"模試 {dt.datetime.now():%Y%m%d-%H%M}",
            questions=questions,
            started_at=dt.datetime.now(),
            year_mode=year_mode,
        )
    session: Optional[ExamSession] = st.session_state.get("exam_session")
    if session is None:
        return
    st.subheader(session.name)
    if st.session_state["settings"].get("timer", True):
        elapsed = dt.datetime.now() - session.started_at
        remaining = max(0, 120 * 60 - int(elapsed.total_seconds()))
        minutes, seconds = divmod(remaining, 60)
        st.info(f"残り時間: {minutes:02d}:{seconds:02d}")
    responses = {}
    correct_count = 0
    for qid in session.questions:
        row = df[df["id"] == qid]
        if row.empty:
            continue
        row = row.iloc[0]
        st.markdown(f"### {row['year']}年 問{row['q_no']}")
        st.write(row["question"])
        choice = st.radio(
            f"回答: {qid}",
            list(range(1, 5)),
            key=f"exam_{qid}",
            horizontal=True,
        )
        responses[qid] = choice
        if row.get("correct") == choice:
            correct_count += 1
    if st.button("採点する"):
        score = correct_count
        st.success(f"得点: {score} 点")
        st.progress(score / max(len(session.questions), 1))
        for qid, choice in responses.items():
            row = df[df["id"] == qid].iloc[0]
            is_correct = row.get("correct") == choice
            db.record_attempt(qid, choice, is_correct, seconds=0, mode="exam")
        st.session_state["exam_session"] = None


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
            payload = sm2_update(row, grade)
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
    st.markdown("### (1) ファイル選択")
    uploaded_files = st.file_uploader(
        "設問・解答ファイルを選択 (CSV/XLSX/ZIP)",
        type=["csv", "xlsx", "xls", "zip"],
        accept_multiple_files=True,
    )
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
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
