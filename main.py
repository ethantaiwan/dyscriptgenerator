from typing import Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
import re
from typing import List# ========= FastAPI 基本設定 =========

app = FastAPI(title="Script Generator (Text Mode)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
DEFAULT_MODEL ="gpt-5.1-mini"
#DEFAULT_MODEL ="gpt-o4-mini"

# ========= Pydantic 請求與回應 =========

class GenerateScriptRequest(BaseModel):
    brand: str = Field(..., description="品牌或公司名稱")
    topic: str = Field(..., description="影片主題")
    video_type: str = Field(..., description="影片類型")
    platform: str = Field(..., description="曝光平台")
    aspect_ratio: str = Field(..., description="影片尺寸比例")
    video_techniques: str = Field(..., description="視覺風格")
    scene_count: int = Field(..., ge=2, le=8, description="分鏡數量")
    tone: Optional[str] = Field(default="自然、溫暖、貼近日常口語")
class TextResult(BaseModel):
    result: str

class ExtractRequest(BaseModel):
    text: str  # 放 /generate-script 回傳的純文字腳本（result）

class ScenePrompt(BaseModel):
    scene_no: int
    prompt: str

class ExtractResponse(BaseModel):
    prompts: List[ScenePrompt]
# ====== Robust 解析器：從純文字中抓出每個 Scene 的 image_prompt ======
IMAGE_PROMPT_KEYS = [
    r"image[_\s-]*prompt",             # image_prompt / image prompt / image-prompt
    r"影像提示", r"生圖提示", r"生成圖提示",  # 若你未來改成中文標
]

def _cleanup_prompt(s: str) -> str:
    # 去掉項目符號、兩側引號（中/英）、多餘空白與換行
    s = re.sub(r'^\s*[-•\u2022]\s*', '', s.strip())
    s = s.strip('「」"“”').strip()
    # 把多行合併為單行（文生圖模型通常比較喜歡單行）
    s = re.sub(r'\s*\n\s*', ' ', s)
    s = re.sub(r'\s{2,}', ' ', s)
    return s.strip()

def extract_image_prompts(text: str) -> List[ScenePrompt]:
    prompts: List[ScenePrompt] = []

    # 先依 Scene 區塊切開；Scene 1…(到下一個 Scene 或結尾)
    scene_blocks = list(re.finditer(
        r"(Scene\s*(\d+)[\s\S]*?)(?=\nScene\s*\d+\b|\Z)", text, flags=re.IGNORECASE
    ))

    for m in scene_blocks:
        block = m.group(1)
        try:
            scene_no = int(m.group(2))
        except Exception:
            # 若 Scene 標題格式跑掉，就給連號
            scene_no = len(prompts) + 1

        # 在該區塊內找 image_prompt 標題行，並抓後面的內容直到下一個空白段落/下一個 Scene
        key_pat = "|".join(IMAGE_PROMPT_KEYS)
        im = re.search(
            rf"(?i)({key_pat}).*?\n"              # image_prompt 標題行（容忍括號/說明文字）
            r"(?:\s*[-•]?\s*)?"                   # 可選的項目符號
            r"(.+?)"                              # 實際提示內容
            r"(?=\n\s*\n|\nScene\s*\d+\b|\Z)",    # 遇到空白段落、下一個 Scene 或檔尾即停止
            block,
            flags=re.DOTALL
        )

        if im:
            raw = im.group(2).strip()
            prompt = _cleanup_prompt(raw)
            if prompt:
                prompts.append(ScenePrompt(scene_no=scene_no, prompt=prompt))
                continue

        # 後備：若沒標到 image_prompt，嘗試用「7) ...」行的下一行
        im2 = re.search(
            r"(?i)\n\s*7\)\s*[^\n]*?\n(.+?)(?=\n\s*\n|\nScene\s*\d+\b|\Z)",
            block, flags=re.DOTALL
        )
        if im2:
            prompt = _cleanup_prompt(im2.group(1))
            if prompt:
                prompts.append(ScenePrompt(scene_no=scene_no, prompt=prompt))

    # 以 scene_no 排序
    prompts.sort(key=lambda x: x.scene_no)
    return prompts


# ========= 系統提示（修正你提供版本的標點與格式） =========
#SYSTEM_PROMPT = (
#    "你是一位專業的短影音腳本與文案創作者，擅長為品牌量身打造多分鏡的短影音腳本。\n"
#    "請依據輸入的品牌主題、影片類型、曝光平台、影片尺寸與視覺風格，自動產生指定數量的分鏡（Scene 1～Scene N；由使用者給定）。\n"
#    "每個分鏡需清楚呈現：\n"
#    "1) 畫面／鏡頭描述（可含景別、運鏡、主體與背景）。\n"
#    "2) 旁白（口語、精準、吸睛）。\n"
#    "3) 情緒／氛圍標籤（2～4 組關鍵詞）。\n"
#    "4) 影片類型標籤（沿用指定類型）。\n"
#    "5) 視覺風格建議（沿用指定風格，可補充光影／質感）。\n"
#    "6) 拍攝技巧（光線、鏡頭、節奏、聲音設計等）。\n\n"
#    "此外，請為『每個分鏡』另外產出兩段專用提示詞：\n"
#    "【image_prompt】：可直接用於文生圖模型，需包含主體、場景、關鍵視覺元素、相機與鏡頭感、構圖、光線、質地、配色與風格關鍵字；避免含有文字浮水印或品牌名。\n"
#    "【video_prompt】：用於影片生成模型（如 Kling）在進行 start-end frame 模式時，能讓上一個分鏡（Scene X）順暢過渡到下一個分鏡（Scene X+1）。\n"
#    "video_prompt 必須包含：\n"
#    "  ‧ 上一個畫面的延續（姿勢、鏡位、光線不跳動）。\n"
#    "  ‧ 邏輯性的動作或鏡頭移動（如：緩慢推進、輕微平移、自然肢體動作）。\n"
#    "  ‧ 必須避免任何『瞬間改變、跳切、姿勢瞬間轉換、光線或構圖突變』。\n"
#    "  ‧ 請以『過渡敘述』方式撰寫，而非最終畫面模板。\n\n"
#    "請同時輸出一段 storyboard_text（純文字整段腳本，方便複製分享）。\n"
#    "語言請使用台灣人習慣的繁體中文（全形標點）。\n"
#)


SYSTEM_PROMPT = (
    "你是一位專業的短影音腳本與視覺過渡設計師，擅長為文生圖與文生影片模型（如 Stable Diffusion、Kling）提供高品質的影像敘事腳本。\n\n"

    "【語言要求】\n"
    "所有輸出必須使用台灣人習慣的繁體中文，不得混用英文。\n"
    "包含 image_prompt 與 video_prompt 全部都要是純中文描述。\n\n"

    "【開頭總結段落】\n"
    "在輸出腳本最前方，請依下列格式產出一段開頭資訊：\n"
    "「以下為依據您提供的條件（品牌：{品牌}、主題：{主題}、平台：{平台}、尺寸：{尺寸}、影片類型：{類型}、視覺風格：{風格}、語氣：{語氣}）所設計的 {場景數} 分鏡腳本。」\n\n"

    "【每個分鏡必須包含的元素】\n"
    "1) 畫面／鏡頭描述（場景、主體、姿勢、背景、光線、動作、鏡頭移動）。\n"
    "2) 旁白（口語化、自然、吸睛）。\n"
    "3) 情緒／氛圍標籤（2～4 組）。\n"
    "4) 影片類型標籤。\n"
    "5) 視覺風格建議（光線、色調、質感）。\n"
    "6) 拍攝技巧（光線、鏡頭、節奏、聲音）。\n\n"

    "【image_prompt 規範 】\n"
    "請為每個分鏡產出 image_prompt：\n"
    "‧ 全部使用繁體中文。\n"
    "‧ 不得出現英文。\n"
    "‧ 不得含有文字、品牌名、浮水印。\n"
    "‧ 詳細描述畫面主體、姿勢、環境、光線、構圖、視覺質地、色調。\n"
    "‧ 單段落輸出，不使用列點。\n\n"

    "【video_prompt 規範 】\n"
    "請比較：\n"
    "‧ Scene X 的 image_prompt（上一張照片）\n"
    "‧ Scene X+1 的 image_prompt（下一張照片）\n"
    "並根據兩張照片的「視覺差異」，輸出一段能讓 Kling 自然補間的過渡描述。\n\n"
    "video_prompt 必須遵守：\n"
    "‧ 僅描述畫面過渡，不描述劇情。\n"
    "‧ 延續上一張照片的主體位置、姿勢、光線，不可跳動。\n"
    "‧ 描述鏡頭運動方式（緩慢推進、平移、拉遠、拉近）。\n"
    "‧ 描述光線如何從 A 過渡到 B（室內→戶外、暖光→夕陽）。\n"
    "‧ 描述背景如何轉換（桌面→公園、照片牆→樹影）。\n"
    "‧ 描述主體如何平滑接到下一張照片的構圖（手的角度、焦點位置）。\n"
    "‧ 整段使用繁體中文，不可出現任何英文。\n\n"

    "【最終輸出】\n"
    "請在最後產生 storyboard_text（純文字整段腳本）。\n"
)



# ========= 建立使用者提示 =========

#def build_user_prompt(payload: GenerateScriptRequest) -> str:
#    return (
#        f"品牌：{payload.brand}\n"
#        f"影片主題：{payload.topic}\n"
#        f"曝光平台：{payload.platform}\n"
#        f"影片尺寸：{payload.aspect_ratio}\n"
#        f"影片類型：{payload.video_type}\n"
#        f"視覺風格：{payload.video_techniques}\n"
#        f"語氣／口吻：{payload.tone}\n"
#        f"分鏡數量：{payload.scene_count} 個場景\n\n"
#        "請依上述條件產生多分鏡腳本，每個分鏡都需包含 image_prompt 與 video_prompt。\n"
#        "image_prompt 不可含有任何文字或品牌名，以避免生圖出現浮水印。\n"
#        "video_prompt 用於影片 start–end frame 過渡，請確保內容敘述能讓上一場景與下一場景銜接順暢。\n"
#        "最後請輸出 storyboard_text（純文字版完整腳本）。"
#    )
def build_user_prompt(payload: GenerateScriptRequest) -> str:
    return (
        f"品牌：{payload.brand}\n"
        f"影片主題：{payload.topic}\n"
        f"曝光平台：{payload.platform}\n"
        f"影片尺寸：{payload.aspect_ratio}\n"
        f"影片類型：{payload.video_type}\n"
        f"視覺風格：{payload.video_techniques}\n"
        f"語氣／口吻：{payload.tone}\n"
        f"分鏡數量：{payload.scene_count} 個場景\n\n"
        "請依照上述條件與 SYSTEM_PROMPT 的規範產生完整腳本，"
        "每個分鏡都需包含 image_prompt 與 video_prompt。"
    )


# ========= API: 產生腳本 =========
@app.post("/generate-script", response_model=TextResult)
def generate_script(req: GenerateScriptRequest):
    try:
        user_prompt = build_user_prompt(req)
        completion = client.chat.completions.create(  # 不要帶 temperature（有些模型不支援）
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        content_text = (completion.choices[0].message.content or "").strip()
        if not content_text:
            raise HTTPException(status_code=502, detail="OpenAI returned empty content")
        return TextResult(result=content_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")
# ========= 健康檢查 =========

@app.get("/")
def root():
    return {"ok": True, "service": "Script-to-Images API", "version": "1.0.0"}

@app.post("/extract-image-prompts", response_model=ExtractResponse)
def extract_image_prompts_endpoint(req: ExtractRequest):
    try:
        prompts = extract_image_prompts(req.text)
        if not prompts:
            raise HTTPException(status_code=422, detail="找不到 image_prompt 段落，請確認腳本格式。")
        return ExtractResponse(prompts=prompts)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parse error: {e}")
