import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

# ========= FastAPI 基本設定 =========
app = FastAPI(title="Script-to-Images API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 上線後可換成你的網域
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= OpenAI Client =========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
#OPENAI_API_KEY  = "sk-uvDM4i9hr1uwsw40LKtRXf8YmuIHCso20rjNCXNumvT3BlbkFJN78QYUUPerFjwHRP7dtJy5lKMcicMH_L6Kuht_1R0A"
if not OPENAI_API_KEY:
    # 在 Render 上到「Environment」加一個 OPENAI_API_KEY
    raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=OPENAI_API_KEY)

# 你也可以用環境變數覆寫 Model
#DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
DEFAULT_MODEL ="gpt-5-mini"
# ========= Pydantic 請求與回應 =========
class GenerateScriptRequest(BaseModel):
    brand: str = Field(..., description="品牌或公司名稱")
    topic: str = Field(..., description="影片主題（你在前端的『主題』）")
    video_type: str = Field(..., description="影片類型（如：一鏡到底、ASMR風格、手持紀錄感、慢動作氛圍、Split Screen 分割畫面、延遲攝影、光影敘事、蒙太奇剪接 等）")
    platform: str = Field(..., description="曝光平台（如 IG Reels / YouTube Shorts / 抖音 等）")
    aspect_ratio: str = Field(..., description="影片尺寸比例（如 9:16 / 16:9 / 1:1 / 3:4 / 4:3）")
    visual_style: str = Field(..., description="視覺風格（如 寫實照片 / 3D 動畫 / 日式手繪 / 立體黏土 / 剪紙風格 等）")
    # 可選：希望口吻/語氣
    tone: Optional[str] = Field(default="自然、溫暖、貼近日常口語")

class Scene(BaseModel):
    scene_no: int
    title: str
    shot_description: str
    voiceover: str
    mood_tags: List[str]
    video_type: str
    platform: str
    aspect_ratio: str
    visual_style: str
    shooting_tips: str
    image_prompt: str  # 直接拿去生圖的提示詞（對齊 shot_description + 風格 + 技法）

class GenerateScriptResponse(BaseModel):
    brand: str
    topic: str
    platform: str
    aspect_ratio: str
    visual_style: str
    video_type: str
    outro_line: str
    storyboard_text: str  # 方便你同時顯示純文字腳本（可直接貼到審稿/簡報）

# ========= 系統提示（修正你提供版本的標點與格式） =========
SYSTEM_PROMPT = (
    "你是一位專業的短影音腳本與文案創作者，擅長為品牌量身打造四分鏡的短影音腳本。\n"
    "請依據輸入的品牌主題、影片類型、曝光平台、影片尺寸與視覺風格，設計四個分鏡（Scene 1～4）。\n"
    "每個分鏡需清楚呈現：\n"
    "1) 畫面/鏡頭描述（可含景別、運鏡、主體與背景）\n"
    "2) 旁白（口語、精準、吸睛）\n"
    "3) 情緒/氛圍標籤（2～4 組關鍵詞）\n"
    "4) 影片類型標籤（沿用指定類型）\n"
    "5) 視覺風格建議（沿用指定風格，可補充光影/質感）\n"
    "6) 拍攝技巧（光線、鏡頭、節奏、聲音設計等）\n"
    "此外，請為『每個分鏡』產出一段可直接用於文生圖模型的 image_prompt：\n"
    "需包含主體、場景、關鍵視覺元素、相機與鏡頭感、構圖、光線、材質、配色與風格關鍵字；避免含有文字水印、Logo。\n"
    "語言請使用台灣人習慣的繁體中文（全形標點）。\n"
)

# ========= JSON Schema（要求模型回傳嚴格 JSON） =========

# ========= 建立使用者提示 =========
def build_user_prompt(payload: GenerateScriptRequest) -> str:
    return (
        f"品牌：{payload.brand}\n"
        f"影片主題：{payload.topic}\n"
        f"曝光平台：{payload.platform}\n"
        f"影片尺寸：{payload.aspect_ratio}\n"
        f"影片類型：{payload.video_type}\n"
        f"視覺風格：{payload.visual_style}\n"
        f"語氣/口吻：{payload.tone}\n\n"
        "請依上述條件產出四分鏡腳本，並確保每個分鏡都含有可直接生成圖片的 image_prompt。\n"
        "image_prompt 不要包含任何品牌名或文字元素，以免生圖出現浮水印或文字。\n"
        "結尾請加入「謝謝使用錨點影音的服務！」作為 outro_line。\n"
        "另外，請同時輸出一段 storyboard_text（純文字整段腳本，方便複製分享）。"
    )

# ========= API: 產生腳本 =========
@app.post("/generate-script", response_model=GenerateScriptResponse)
def generate_script(req: GenerateScriptRequest):
    try:
        user_prompt = build_user_prompt(req)

        # ✅ 正確用法：新版 OpenAI SDK 語法
        completion = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
        )

        # 取出生成文字
        content_text = completion.choices[0].message.content.strip()
        return {"result": content_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")

@app.get("/")
def root():
    return {"ok": True, "msg": "Script Generator (text mode) is running."}
    # 將 JSON 轉回 Pydantic 物件
    #import json
    #try:
    #    data = json.loads(content_text)
    #    return GenerateScriptResponse(**data)
    #except Exception as e:
    #    raise HTTPException(status_code=500, detail=f"JSON parse error: {e}")

# ========= 健康檢查 =========
@app.get("/")
def root():
    return {"ok": True, "service": "Script-to-Images API", "version": "1.0.0"}
