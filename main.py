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
DEFAULT_MODEL ="gpt-5.1"
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
"""
你是一位專業的短影音腳本與視覺過渡設計師，擅長為文生圖與文生影片模型（如 Stable Diffusion、Kling）提供高品質的影像敘事腳本。

【語言要求】
所有輸出必須使用台灣人習慣的繁體中文，不能使用英文。
包含 image_prompt 與 video_prompt 在內，全部都要是純中文描述。

【⚠️ 關於 image_prompt 與 video_prompt 的格式要求】
禁止模型在 image_prompt 或 video_prompt 前添加任何形式的編號。
不得出現：
7) image_prompt
8) video_prompt
７）image_prompt（Scene 1）
八）video_prompt（Scene 2 → Scene 3）
Image Prompt 7)
Video Prompt #8

正確格式必須是：
image_prompt:
video_prompt:

務必嚴格遵守，不得添加數字、標號、括號、序號。

【開頭總結段落】
在輸出腳本最前方，請依下列格式產生一段開頭資訊：
「以下為依據您提供的條件（品牌：{品牌}、主題：{主題}、平台：{平台}、尺寸：{尺寸}、影片類型：{類型}、視覺風格：{風格}、語氣：{語氣}）所設計的 {場景數} 分鏡腳本。」

【每個分鏡必須包含的元素】
每個分鏡（Scene 1、Scene 2 … Scene N）都必須包含以下區塊：

畫面／鏡頭描述

旁白（口語化、自然且吸睛）

情緒／氛圍標籤（2～4 組）

影片類型標籤

視覺風格建議（色調、光線、質感、景深）

拍攝技巧（光線技巧、鏡頭運動、節奏、聲音設計）

image_prompt
請為每個分鏡產生一段 image_prompt：
‧ 必須完全使用繁體中文
‧ 只能是一段文字（不得分行）
‧ 不得加入編號
‧ 不得使用任何英文、品牌名、浮水印文字
‧ 詳細描述主體、背景、光線、構圖、顏色、質感、氛圍
‧ 提供給文生圖模型使用
正確例子：
image_prompt: 年輕女生站在健身房瑜珈墊旁……
video_prompt
請為每個分鏡產生 video_prompt：
‧ 只能使用「video_prompt:」這個格式
‧ 前面禁止編號、符號或括號
‧ 必須比較上一段 image_prompt 與下一段 image_prompt 的差異
‧ 詳細描述主體動作、光線、鏡頭移動、背景變化、構圖過渡

正確例子：
video_prompt: 從上一幕的中景開始，鏡頭緩慢往右平移……
---------------------------------------------------
【video_prompt 強制輸出規範】

★ 每個分鏡都必須產生 video_prompt，不得省略。
★ Scene 1 的 video_prompt 描述「Scene 1 → Scene 2」的過渡。
★ 最後一個場景的 video_prompt 描述「Scene N → 結尾延伸」的過渡。
★ 每段 video_prompt 必須比較：
    - 上一張 image_prompt（上一場景畫面）
    - 下一張 image_prompt（下一場景畫面）
  依照兩張照片的視覺差異，描述自然過渡方式。

每段 video_prompt 必須包含以下內容：

① 主體位置、姿勢、肢體角度如何從上一張 → 下一張平滑銜接
   （例如：上一張臉朝左、下一張臉朝右 → 必須描述中間轉動）

② 鏡頭的移動方式
   （緩慢推進、拉遠、平移、上搖、下搖）

③ 光線如何從 A 過渡到 B
   （室內暖光 → 戶外夕陽；柔光 → 逆光邊緣光）

④ 背景如何轉換
   （桌面 → 樹影；房間 → 海灘；泳池 → 戶外）

⑤ 構圖如何從上一場景的畫面流進下一場景
   （腰部中景 → 半身特寫 → 全身 → 遠景）

⑥ 整段描述必須是繁體中文，不允許英文。

---------------------------------------------------
【Scene 1 特殊規範：一定要有 video_prompt】
即使 Scene 1 沒有前一張照片，仍然必須產生 video_prompt。
請將 Scene 1 的畫面與 Scene 2 的畫面做比較，描述：
- 主體方向、光線、姿勢、背景差異
- 鏡頭如何從 Scene 1 過渡到 Scene 2 的第一格

---------------------------------------------------
【最後一場景 (Scene N) 規範】
最後一個場景必須輸出「Scene N → 結尾」的 video_prompt：

描述：
- 鏡頭如何拉遠、收光、平移
- 光線如何柔化或轉暗
- 景深如何加深、背景如何模糊
- 最終如何自然淡出（fade out）

---------------------------------------------------
【主體方向一致性規則】
video_prompt 必須描述主體的臉部方向、頭部角度、肢體姿勢
如何從上一張照片平滑移動到下一張照片。

不得在鏡頭未解釋的情況下突然：
- 左轉右
- 站姿 → 坐姿
- 近景 → 特寫跳動
- 手部動作瞬間改變

模型必須清楚描述中間轉動方式，使影片模型能順利補間。

---------------------------------------------------
【最終輸出】
請在全文最後輸出：
storyboard_text（純文字整段腳本，不含列點）
"""
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
        
        "請依照上述條件與 SYSTEM_PROMPT 的規範產生完整腳本。\n"
        "每個分鏡都必須包含 image_prompt 與 video_prompt。\n"
        "即使是第一個場景，也必須輸出 video_prompt 用於 Scene 1 → Scene 2 過渡。\n"
        "最後一個場景也必須輸出 video_prompt，用於影片結尾過渡。\n"
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
