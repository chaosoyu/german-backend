from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import json
import os
from sqlalchemy import create_engine, Column, Integer, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# ==========================================
# 1. 資料庫設定 (支援雲端 PostgreSQL 與本地 SQLite)
# ==========================================
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./german_quiz.db")

# 修正部分平台舊格式網址前綴
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# 區分 SQLite 與雲端 PostgreSQL 的連線設定 (加入 pool_pre_ping 防止雲端休眠斷線)
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class NounModel(Base):
    __tablename__ = "nouns"
    id = Column(Integer, primary_key=True, index=True)
    word = Column(String, unique=True, index=True)
    gender = Column(String)
    plural = Column(String)
    translation = Column(String)
    correct_count = Column(Integer, default=0)
    wrong_count = Column(Integer, default=0)

class VerbModel(Base):
    __tablename__ = "verbs"
    id = Column(Integer, primary_key=True, index=True)
    infinitive = Column(String, unique=True, index=True)
    translation = Column(String)
    praesens = Column(JSON)
    correct_count = Column(Integer, default=0)
    wrong_count = Column(Integer, default=0)

class SentenceModel(Base):
    __tablename__ = "sentences"
    id = Column(Integer, primary_key=True, index=True)
    german = Column(String, unique=True)
    type = Column(String)
    translation = Column(String)
    words = Column(JSON)
    correct_count = Column(Integer, default=0)
    wrong_count = Column(Integer, default=0)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================================
# 2. FastAPI 與 Gemini 設定 (改由環境變數讀取金鑰)
# ==========================================
app = FastAPI(title="德語題庫解析引擎 (雲端部署版)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🌟 從雲端環境變數讀取 API Key，絕不將金鑰明文寫在程式碼中
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

class TextInput(BaseModel):
    text: str

class QuizResult(BaseModel):
    type: str
    item_id: int
    is_correct: bool

PROMPT_TEMPLATE = """
你現在是一位專業的德語老師與語言學家。請分析提供的德文內容，提取重要的名詞、動詞與句子，並自動轉換成德語學習題庫的 JSON 格式。
【規則】
1. nouns: 提取約 30 到 60 個核心名詞，包含 word, gender (der/die/das), plural, translation。
2. verbs: 提取約 30 到 50 個核心動詞，包含 infinitive, translation, 以及 praesens (現在式六大人稱變化)。
3. sentences: 提取約 10 到 30 個實用句子，包含 german, type (Hauptsatz/W-Frage/Ja/Nein-Frage), translation, 以及 words 陣列(打散的單字)。
👉 警告：請控制輸出長度！為避免內容遭系統強制截斷，請精選最重要的單字，並確保最後的 JSON 括號完全閉合！
必須嚴格遵守以下 JSON 格式：
{"nouns": [{"word": "Apfel", "gender": "der", "plural": "Äpfel", "translation": "蘋果"}], "verbs": [{"infinitive": "sprechen", "translation": "說", "praesens": {"ich": "spreche", "du": "sprichst", "er_sie_es": "spricht", "wir": "sprechen", "ihr": "sprecht", "sie_Sie": "sprechen"}}], "sentences": [{"german": "Ich lerne Deutsch.", "type": "Hauptsatz", "translation": "我學德文。", "words": ["Ich", "Deutsch", "lerne"]}]}
"""

# ==========================================
# 3. 核心 API 路由
# ==========================================
@app.get("/api/quizzes")
def get_all_quizzes(db: Session = Depends(get_db)):
    nouns = db.query(NounModel).all()
    verbs = db.query(VerbModel).all()
    sentences = db.query(SentenceModel).all()
    return {
        "nouns": [{"id": n.id, "word": n.word, "gender": n.gender, "plural": n.plural, "translation": n.translation, "correct_count": n.correct_count, "wrong_count": n.wrong_count} for n in nouns],
        "verbs": [{"id": v.id, "infinitive": v.infinitive, "translation": v.translation, "praesens": v.praesens, "correct_count": v.correct_count, "wrong_count": v.wrong_count} for v in verbs],
        "sentences": [{"id": s.id, "german": s.german, "type": s.type, "translation": s.translation, "words": s.words, "correct_count": s.correct_count, "wrong_count": s.wrong_count} for s in sentences]
    }

@app.post("/api/record-result")
def record_result(result: QuizResult, db: Session = Depends(get_db)):
    model_map = {"noun": NounModel, "verb": VerbModel, "sentence": SentenceModel}
    model = model_map.get(result.type)
    if not model: return {"error": "Invalid type"}
    
    item = db.query(model).filter(model.id == result.item_id).first()
    if item:
        if result.is_correct:
            item.correct_count += 1
        else:
            item.wrong_count += 1
        db.commit()
    return {"message": "已更新學習紀錄"}

def save_quiz_to_db(quiz_data: dict, db: Session):
    for n in quiz_data.get("nouns", []):
        if not db.query(NounModel).filter(NounModel.word == n["word"]).first():
            db.add(NounModel(word=n["word"], gender=n["gender"], plural=n["plural"], translation=n["translation"], correct_count=n.get("correct_count", 0), wrong_count=n.get("wrong_count", 0)))
    for v in quiz_data.get("verbs", []):
        if not db.query(VerbModel).filter(VerbModel.infinitive == v["infinitive"]).first():
            db.add(VerbModel(infinitive=v["infinitive"], translation=v["translation"], praesens=v["praesens"], correct_count=v.get("correct_count", 0), wrong_count=v.get("wrong_count", 0)))
    for s in quiz_data.get("sentences", []):
        if not db.query(SentenceModel).filter(SentenceModel.german == s["german"]).first():
            db.add(SentenceModel(german=s["german"], type=s["type"], translation=s["translation"], words=s["words"], correct_count=s.get("correct_count", 0), wrong_count=s.get("wrong_count", 0)))
    db.commit()

@app.post("/api/generate-quiz")
def generate_quiz(input_data: TextInput, db: Session = Depends(get_db)):
    quiz_data = call_gemini([PROMPT_TEMPLATE, f"【要分析的德文內容】\n{input_data.text}"])
    save_quiz_to_db(quiz_data, db)
    return {"message": "✅ 文字解析成功並已寫入資料庫！"}

@app.post("/api/upload-file")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        file_bytes = await file.read()
        quiz_data = call_gemini([PROMPT_TEMPLATE, {"mime_type": file.content_type, "data": file_bytes}])
        save_quiz_to_db(quiz_data, db)
        return {"message": "✅ 圖片/PDF 辨識成功並已寫入資料庫！"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/quizzes")
def clear_database(db: Session = Depends(get_db)):
    db.query(NounModel).delete()
    db.query(VerbModel).delete()
    db.query(SentenceModel).delete()
    db.commit()
    return {"message": "🗑️ 資料庫已清空"}

@app.post("/api/import-db")
def import_to_db(quiz_data: dict, db: Session = Depends(get_db)):
    try:
        save_quiz_to_db(quiz_data, db)
        return {"message": "✅ 題庫匯入成功！"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"匯入失敗: {str(e)}")

def call_gemini(contents):
    model = genai.GenerativeModel('gemini-3.5-flash-lite') 
    response = model.generate_content(contents, generation_config={"response_mime_type": "application/json", "max_output_tokens": 8192})
    return json.loads(response.text.strip())