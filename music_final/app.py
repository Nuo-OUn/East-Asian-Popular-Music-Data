from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


BASE_DIR = Path(__file__).resolve().parent
PCA_CSV = BASE_DIR / "east_asian_pca_3d.csv"
FEATURES_CSV = BASE_DIR / "east_asian_pop_features.csv"
LOADINGS_CSV = BASE_DIR / "pca_loadings.csv"

# 雷达图所用的 6 个 [0, 1] 归一化声学特征
# (不含 Loudness/Tempo/Key/Mode 等不在 0-1 范围的指标)
RADAR_FEATURES = [
    "Danceability",
    "Energy",
    "Valence",
    "Acousticness",
    "Speechiness",
    "Liveness",
]


# ==========================================
# 地区分类(C-pop / J-pop / K-pop)
# 策略: 先查显式映射表; 查不到则用 Unicode 字符规则兜底
# ==========================================
REGION_C = "C-pop"
REGION_J = "J-pop"
REGION_K = "K-pop"
REGION_OTHER = "Other"

# 三大地区的固定显示颜色 (暗底友好、彼此区分度高)
REGION_COLORS: dict[str, str] = {
    REGION_C: "#e15759",  # 暖红
    REGION_J: "#76b7b2",  # 青蓝
    REGION_K: "#f1a93b",  # 暖黄
    REGION_OTHER: "#8b8f99",  # 灰
}

# 显式映射表: 数据集里常见的东亚歌手 (英文/罗马音/原文 多写法均覆盖)
# key 一律小写, 用于大小写不敏感匹配
ARTIST_REGION_MAP: dict[str, str] = {
    # ---------- C-pop ----------
    "jj lin": REGION_C, "林俊杰": REGION_C, "lin junjie": REGION_C,
    "jay chou": REGION_C, "周杰伦": REGION_C, "周杰倫": REGION_C,
    "eason chan": REGION_C, "陈奕迅": REGION_C, "陳奕迅": REGION_C,
    "jolin tsai": REGION_C, "蔡依林": REGION_C,
    "g.e.m.": REGION_C, "g.e.m": REGION_C, "邓紫棋": REGION_C, "鄧紫棋": REGION_C,
    "wang leehom": REGION_C, "leehom wang": REGION_C, "王力宏": REGION_C,
    "jolin": REGION_C,
    "stefanie sun": REGION_C, "孙燕姿": REGION_C, "孫燕姿": REGION_C,
    "a-mei": REGION_C, "amei": REGION_C, "张惠妹": REGION_C, "張惠妹": REGION_C,
    "faye wong": REGION_C, "王菲": REGION_C,
    "andy lau": REGION_C, "刘德华": REGION_C, "劉德華": REGION_C,
    "jacky cheung": REGION_C, "张学友": REGION_C, "張學友": REGION_C,
    "leslie cheung": REGION_C, "张国荣": REGION_C, "張國榮": REGION_C,
    "anita mui": REGION_C, "梅艳芳": REGION_C, "梅艷芳": REGION_C,
    "teresa teng": REGION_C, "邓丽君": REGION_C, "鄧麗君": REGION_C,
    "mayday": REGION_C, "五月天": REGION_C,
    "sodagreen": REGION_C, "苏打绿": REGION_C, "蘇打綠": REGION_C,
    "f.i.r.": REGION_C, "fir": REGION_C, "飞儿乐团": REGION_C,
    "khalil fong": REGION_C, "方大同": REGION_C,
    "li ronghao": REGION_C, "李荣浩": REGION_C, "李榮浩": REGION_C,
    "hua chenyu": REGION_C, "华晨宇": REGION_C, "華晨宇": REGION_C,
    "li yuchun": REGION_C, "chris lee": REGION_C, "李宇春": REGION_C,
    "tia ray": REGION_C, "袁娅维": REGION_C, "袁婭維": REGION_C,
    "tanya chua": REGION_C, "蔡健雅": REGION_C,
    "fish leong": REGION_C, "梁静茹": REGION_C, "梁靜茹": REGION_C,
    "rainie yang": REGION_C, "杨丞琳": REGION_C, "楊丞琳": REGION_C,
    "show lo": REGION_C, "罗志祥": REGION_C, "羅志祥": REGION_C,
    "vae xu": REGION_C, "许嵩": REGION_C, "許嵩": REGION_C,
    "joker xue": REGION_C, "薛之谦": REGION_C, "薛之謙": REGION_C,
    "jam hsiao": REGION_C, "萧敬腾": REGION_C, "蕭敬騰": REGION_C,
    "yoga lin": REGION_C, "林宥嘉": REGION_C,
    "crowd lu": REGION_C, "卢广仲": REGION_C, "盧廣仲": REGION_C,
    "namewee": REGION_C, "黄明志": REGION_C, "黃明志": REGION_C,

    # ---------- K-pop ----------
    "bts": REGION_K, "방탄소년단": REGION_K, "防弹少年团": REGION_K,
    "blackpink": REGION_K, "블랙핑크": REGION_K,
    "exo": REGION_K, "엑소": REGION_K,
    "twice": REGION_K, "트와이스": REGION_K,
    "red velvet": REGION_K, "레드벨벳": REGION_K,
    "girls' generation": REGION_K, "snsd": REGION_K, "소녀시대": REGION_K, "少女时代": REGION_K,
    "big bang": REGION_K, "bigbang": REGION_K, "빅뱅": REGION_K,
    "got7": REGION_K, "갓세븐": REGION_K,
    "seventeen": REGION_K, "세븐틴": REGION_K,
    "nct": REGION_K, "nct 127": REGION_K, "nct dream": REGION_K, "nct u": REGION_K, "엔시티": REGION_K,
    "stray kids": REGION_K, "스트레이 키즈": REGION_K,
    "ateez": REGION_K, "에이티즈": REGION_K,
    "txt": REGION_K, "tomorrow x together": REGION_K, "투모로우바이투게더": REGION_K,
    "enhypen": REGION_K, "엔하이픈": REGION_K,
    "ive": REGION_K, "아이브": REGION_K,
    "newjeans": REGION_K, "뉴진스": REGION_K,
    "le sserafim": REGION_K, "르세라핌": REGION_K,
    "(g)i-dle": REGION_K, "g-idle": REGION_K, "여자아이들": REGION_K,
    "itzy": REGION_K, "있지": REGION_K,
    "aespa": REGION_K, "에스파": REGION_K,
    "mamamoo": REGION_K, "마마무": REGION_K,
    "iu": REGION_K, "아이유": REGION_K,
    "psy": REGION_K, "싸이": REGION_K,
    "g-dragon": REGION_K, "gd": REGION_K, "지드래곤": REGION_K,
    "taeyang": REGION_K, "태양": REGION_K,
    "jennie": REGION_K, "제니": REGION_K,
    "lisa": REGION_K, "리사": REGION_K,
    "rosé": REGION_K, "rose": REGION_K, "로제": REGION_K,
    "jisoo": REGION_K, "지수": REGION_K,
    "jungkook": REGION_K, "정국": REGION_K,
    "j-hope": REGION_K, "j hope": REGION_K, "제이홉": REGION_K,
    "jimin": REGION_K, "지민": REGION_K,
    "v": REGION_K,
    "rm": REGION_K, "rapmonster": REGION_K,
    "suga": REGION_K, "agust d": REGION_K,
    "jin": REGION_K,
    "taeyeon": REGION_K, "태연": REGION_K,
    "sunmi": REGION_K, "선미": REGION_K,
    "hyuna": REGION_K, "현아": REGION_K,
    "chungha": REGION_K, "청하": REGION_K,
    "zico": REGION_K, "지코": REGION_K,
    "epik high": REGION_K, "에픽하이": REGION_K,
    "akmu": REGION_K, "akdong musician": REGION_K, "악동뮤지션": REGION_K,
    "dean": REGION_K, "딘": REGION_K,
    "crush": REGION_K, "크러쉬": REGION_K,
    "hyolyn": REGION_K, "효린": REGION_K,
    "heize": REGION_K, "헤이즈": REGION_K,
    "monsta x": REGION_K, "몬스타엑스": REGION_K,
    "shinee": REGION_K, "샤이니": REGION_K,
    "super junior": REGION_K, "슈퍼주니어": REGION_K,
    "tvxq": REGION_K, "tvxq!": REGION_K, "동방신기": REGION_K,
    "2pm": REGION_K, "투피엠": REGION_K,
    "ikon": REGION_K, "아이콘": REGION_K,
    "winner": REGION_K, "위너": REGION_K,
    "day6": REGION_K, "데이식스": REGION_K,
    "wanna one": REGION_K, "워너원": REGION_K,
    "iz*one": REGION_K, "izone": REGION_K, "아이즈원": REGION_K,
    "loona": REGION_K, "loonatheworld": REGION_K, "이달의 소녀": REGION_K,
    "kep1er": REGION_K, "케플러": REGION_K,
    "fifty fifty": REGION_K,
    "zerobaseone": REGION_K, "zb1": REGION_K, "제로베이스원": REGION_K,
    "bibi": REGION_K, "비비": REGION_K,
    "younha": REGION_K, "윤하": REGION_K,
    "10cm": REGION_K,
    "paul kim": REGION_K, "폴킴": REGION_K,
    "ben": REGION_K,

    # ---------- J-pop ----------
    "kenshi yonezu": REGION_J, "yonezu kenshi": REGION_J, "米津玄師": REGION_J, "米津玄师": REGION_J,
    "hikaru utada": REGION_J, "utada hikaru": REGION_J, "utada": REGION_J, "宇多田ヒカル": REGION_J, "宇多田光": REGION_J,
    "yoasobi": REGION_J, "ヨアソビ": REGION_J,
    "ado": REGION_J, "アド": REGION_J,
    "lisa": REGION_J,  # 注: 与 BLACKPINK Lisa 重名,保守归 J-pop 因数据集中常见; 用户可手动覆盖
    "aimer": REGION_J, "エメ": REGION_J,
    "official髭男dism": REGION_J, "official hige dandism": REGION_J, "ヒゲダン": REGION_J,
    "king gnu": REGION_J, "キングヌー": REGION_J,
    "mrs. green apple": REGION_J, "mrs green apple": REGION_J, "ミセス": REGION_J,
    "vaundy": REGION_J, "ヴァウンディ": REGION_J,
    "fujii kaze": REGION_J, "kaze fujii": REGION_J, "藤井風": REGION_J,
    "yorushika": REGION_J, "ヨルシカ": REGION_J,
    "zutomayo": REGION_J, "ずっと真夜中でいいのに。": REGION_J,
    "eve": REGION_J, "イヴ": REGION_J,
    "kana-boon": REGION_J, "kanaboon": REGION_J,
    "radwimps": REGION_J, "ラッドウィンプス": REGION_J,
    "one ok rock": REGION_J, "ワンオクロック": REGION_J,
    "spitz": REGION_J, "スピッツ": REGION_J,
    "mr.children": REGION_J, "mr children": REGION_J, "ミスチル": REGION_J,
    "back number": REGION_J, "バックナンバー": REGION_J,
    "sekai no owari": REGION_J, "セカオワ": REGION_J,
    "perfume": REGION_J, "パフューム": REGION_J,
    "babymetal": REGION_J, "ベビーメタル": REGION_J,
    "akb48": REGION_J,
    "nogizaka46": REGION_J, "乃木坂46": REGION_J,
    "keyakizaka46": REGION_J, "欅坂46": REGION_J,
    "sakurazaka46": REGION_J, "櫻坂46": REGION_J,
    "ayumi hamasaki": REGION_J, "浜崎あゆみ": REGION_J,
    "namie amuro": REGION_J, "安室奈美恵": REGION_J,
    "arashi": REGION_J, "嵐": REGION_J, "岚": REGION_J,
    "smap": REGION_J,
    "kinki kids": REGION_J, "キンキキッズ": REGION_J,
    "v6": REGION_J,
    "exile": REGION_J, "エグザイル": REGION_J,
    "lisa (japan)": REGION_J,
    "supercell": REGION_J, "スーパーセル": REGION_J,
    "ikimonogakari": REGION_J, "いきものがかり": REGION_J,
    "aiko": REGION_J, "あいこ": REGION_J,
    "yui": REGION_J,
    "yuki": REGION_J, "ユキ": REGION_J,
    "minami": REGION_J,
    "miliyah kato": REGION_J, "加藤ミリヤ": REGION_J,
    "yuzu": REGION_J, "ゆず": REGION_J,
    "porno graffitti": REGION_J, "porno graffiti": REGION_J,
    "tatsuro yamashita": REGION_J, "山下達郎": REGION_J,
    "mariya takeuchi": REGION_J, "竹内まりや": REGION_J,
    "anri": REGION_J, "杏里": REGION_J,
    "milet": REGION_J,
    "minami chiba": REGION_J,
    "yuuri": REGION_J, "優里": REGION_J,
    "macaroni enpitsu": REGION_J, "マカロニえんぴつ": REGION_J,
    "saucy dog": REGION_J, "saucy dog": REGION_J,
    "creepy nuts": REGION_J, "クリーピーナッツ": REGION_J,
    "tani yuuki": REGION_J, "谷優祐": REGION_J,
    "imase": REGION_J,
    "atarashii gakko!": REGION_J, "新しい学校のリーダーズ": REGION_J,
    "fujifabric": REGION_J, "フジファブリック": REGION_J,
    "asian kung-fu generation": REGION_J, "akfg": REGION_J,
    "the oral cigarettes": REGION_J,
    "frederic": REGION_J, "フレデリック": REGION_J,
    "polkadot stingray": REGION_J, "ポルカドットスティングレイ": REGION_J,
    "wednesday campanella": REGION_J, "水曜日のカンパネラ": REGION_J,
}


# 用于字符规则兜底的 Unicode 区间判定
def _has_chinese_chars(s: str) -> bool:
    """检测是否含中日共用 CJK 表意字符 (注意: 这并不能区分中文和日文汉字本身)。"""
    return any("\u4e00" <= c <= "\u9fff" for c in s) or any("\u3400" <= c <= "\u4dbf" for c in s)


def _has_japanese_kana(s: str) -> bool:
    """平假名 (3040-309F) + 片假名 (30A0-30FF) — 只在日文中出现。"""
    return any("\u3040" <= c <= "\u309f" for c in s) or any("\u30a0" <= c <= "\u30ff" for c in s)


def _has_korean_hangul(s: str) -> bool:
    """谚文音节 (AC00-D7AF) + 兼容字母 (3130-318F)。"""
    return any("\uac00" <= c <= "\ud7af" for c in s) or any("\u3130" <= c <= "\u318f" for c in s)


def classify_artist_region(artist_name: str) -> str:
    """
    返回该歌手所属地区: C-pop / J-pop / K-pop / Other。
    优先查映射表; 查不到则用 Unicode 字符规则:
      - 含假名 → J-pop
      - 含韩文 → K-pop
      - 含 CJK 汉字(且不含假名/韩文) → 默认 C-pop (因为日文里独立汉字命名的艺人很少, 而中文歌手大量使用)
      - 纯英文 / 罗马音且不在映射表 → Other (用户可手动选歌手时仍可单独显示)
    """
    if not artist_name:
        return REGION_OTHER

    # 1) 显式映射 (大小写不敏感)
    key = artist_name.strip().lower()
    if key in ARTIST_REGION_MAP:
        return ARTIST_REGION_MAP[key]

    # 2) 字符规则兜底
    name = artist_name.strip()
    if _has_japanese_kana(name):
        return REGION_J
    if _has_korean_hangul(name):
        return REGION_K
    if _has_chinese_chars(name):
        # 注意: 没有假名 / 没有韩文 的 CJK 表意文字, 在东亚流行乐数据集里压倒性多数是中文歌手
        # (如 "周杰倫", "陳奕迅"), 因此默认 C-pop
        return REGION_C

    # 3) 纯拉丁字符且不在映射表内
    return REGION_OTHER


def get_region_color(region: str) -> str:
    return REGION_COLORS.get(region, REGION_COLORS[REGION_OTHER])


# ==========================================
# 多歌手解析:把 "['BTS', 'Halsey']" / "BTS, Halsey" / "BTS feat. Halsey"
# 这类组合字段拆成单个歌手列表,从而支持"包含"语义筛选
# ==========================================
# 仅使用"强分隔符"——这些字符极少出现在艺人本名中,可以安全拆分
# 注意:刻意不包含 'and' / 'with' / 'feat.' 等英文连接词,以避免误伤
# 'Earth, Wind and Fire' / 'Mumford & Sons' 这类含连接词的真名艺人。
# Kaggle Spotify 数据集 ~99% 的合作字段是 "['A', 'B']" 或 "A, B" 形式,
# 这两种已被准确覆盖。
_STRONG_SPLIT_RE = re.compile(r"\s*[,;、，；・]\s*")
# feat. / ft. 是另一种可信的合作分隔信号,单独处理(后接空白才视为分隔)
_FEAT_RE = re.compile(r"\s+(?:feat\.?|ft\.?)\s+", flags=re.IGNORECASE)


def parse_artists_field(raw: Any) -> list[str]:
    """
    将任意格式的歌手字段解析为单个歌手列表。
    覆盖典型情况:
      - "['BTS', 'Halsey']"   -> ['BTS', 'Halsey']
      - "BTS, Halsey"         -> ['BTS', 'Halsey']
      - "JJ Lin、王力宏"      -> ['JJ Lin', '王力宏']
      - "BTS feat. Halsey"    -> ['BTS', 'Halsey']
      - "BTS"                 -> ['BTS']
      - "Mumford & Sons"      -> ['Mumford & Sons']  # 不误拆 & 在艺人本名中
      - NaN / ""              -> []
    """
    if raw is None:
        return []
    if isinstance(raw, float) and pd.isna(raw):
        return []

    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return []

    # 1) 优先按 Python 字面量解析(Spotify 数据常见 "['A', 'B']" 形式)
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                cands = [str(x).strip() for x in parsed if str(x).strip()]
                return _dedupe_keep_order(cands)
        except (ValueError, SyntaxError):
            s = s.strip("[]").replace("'", "").replace('"', "")

    # 2) feat. / ft. 视为合作分隔
    s = _FEAT_RE.sub(", ", s)

    # 3) 按强分隔符拆分(逗号、分号、顿号、全角标点等)
    parts = _STRONG_SPLIT_RE.split(s)
    cleaned = [p.strip().strip("'\"") for p in parts if p and p.strip()]
    return _dedupe_keep_order(cleaned)


def _dedupe_keep_order(items: list[str]) -> list[str]:
    """大小写不敏感去重,保留首次出现顺序。"""
    seen: set[str] = set()
    out: list[str] = []
    for p in items:
        key = p.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out


def attach_artist_columns(df: pd.DataFrame, artist_col: str) -> pd.DataFrame:
    """
    在原表上追加这些列:
      - Artists_List:    list[str], 该曲目的所有参与歌手
      - Primary_Artist:  str, 主唱(列表第 0 位), 用于 3D 着色时避免同一首歌重复出现
      - Primary_Region:  str, 主唱所属地区 (C-pop / J-pop / K-pop / Other)
    """
    df = df.copy()
    df["Artists_List"] = df[artist_col].apply(parse_artists_field)
    df["Primary_Artist"] = df["Artists_List"].apply(
        lambda xs: xs[0] if xs else "(Unknown Artist)"
    )
    df["Primary_Region"] = df["Primary_Artist"].apply(classify_artist_region)
    return df


def collect_artists_by_region(artist_pool: list[str]) -> dict[str, list[str]]:
    """
    Bucket the single-artist pool into the four region buckets.
    Returns: {REGION_C: [...], REGION_J: [...], REGION_K: [...], REGION_OTHER: [...]},
    each list preserving the input order.
    """
    buckets: dict[str, list[str]] = {
        REGION_C: [],
        REGION_J: [],
        REGION_K: [],
        REGION_OTHER: [],
    }
    for a in artist_pool:
        r = classify_artist_region(a)
        if r in buckets:
            buckets[r].append(a)
    return buckets


def filter_by_any_artist(df: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
    """
    "包含"语义筛选:任一被选歌手出现在 Artists_List 中即命中。
    大小写不敏感。

    同时附加一列 Display_Artist:
      = 该曲目的 Artists_List 中,第一个落入 selected 集合的歌手,
        **统一回归到 selected 里的规范拼写**(避免同名艺人在数据里有大小写/空格变体
        导致同一人在图例里出现多种颜色)。
    最终保证:图表里出现的歌手类别 ⊆ selected,即"只显示用户已选歌手"。
    """
    if not selected:
        return df.iloc[0:0].copy()

    # lower → 用户提供的规范拼写
    needles_lower_to_canonical: dict[str, str] = {a.lower(): a for a in selected}

    def first_selected(xs: list[str]) -> Optional[str]:
        for a in xs:
            canon = needles_lower_to_canonical.get(a.lower())
            if canon is not None:
                # 关键:返回的是 selected 中的规范拼写,不是数据里的原拼写
                # 这样图例颜色按规范拼写聚合,不会因同名变体被拆成多类
                return canon
        return None

    out = df.copy()
    out["Display_Artist"] = out["Artists_List"].apply(first_selected)
    out = out[out["Display_Artist"].notna()].copy()
    # 给每首被选中的曲目按其 Display_Artist 标注地区,供"按地区聚合"模式使用
    out["Display_Region"] = out["Display_Artist"].apply(classify_artist_region)
    return out


def build_single_artist_pool(df: pd.DataFrame) -> list[str]:
    """
    扁平化所有歌手为单个歌手列表(去重),并按出现频次降序返回。
    检索候选池只包含**单个歌手**,不会出现 'BTS, Halsey' 这种组合项。
    """
    counter: dict[str, int] = {}
    canonical: dict[str, str] = {}
    for xs in df["Artists_List"]:
        for a in xs:
            key = a.lower()
            if key not in canonical:
                canonical[key] = a
            counter[key] = counter.get(key, 0) + 1
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return [canonical[k] for k, _ in items]


def count_per_single_artist(df: pd.DataFrame) -> dict[str, int]:
    """统计每个单歌手在当前数据中出现的曲目数(任一参与即计 1)。"""
    out: dict[str, int] = {}
    for xs in df["Artists_List"]:
        seen: set[str] = set()
        for a in xs:
            key = a.lower()
            if key in seen:
                continue
            seen.add(key)
            out[a] = out.get(a, 0) + 1
    return out


# ==========================================
# 绘图工具
# ==========================================
def make_histogram_row(frame: pd.DataFrame, cols: list[str]) -> go.Figure:
    sub = make_subplots(rows=1, cols=len(cols), subplot_titles=cols)
    for i, col in enumerate(cols, start=1):
        sub.add_trace(
            go.Histogram(
                x=frame[col],
                name=col,
                showlegend=False,
                marker=dict(
                    color="#7c8aa3",
                    line=dict(color="#1a1d24", width=0.6),
                ),
            ),
            row=1,
            col=i,
        )
    sub.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=56, b=24),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color="#d4d6db"),
    )
    sub.update_xaxes(
        showgrid=True,
        gridcolor="rgba(140, 148, 165, 0.15)",
        color="#8b8f99",
    )
    sub.update_yaxes(
        showgrid=True,
        gridcolor="rgba(140, 148, 165, 0.15)",
        color="#8b8f99",
    )
    # 子图标题颜色
    for ann in sub["layout"]["annotations"]:
        ann["font"] = dict(color="#d4d6db", size=13)
    return sub


def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def build_hover_data(
    plot_df: pd.DataFrame,
    color_col: str,
    hover_numeric: list[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {c: True for c in hover_numeric if c in plot_df.columns}
    for geo in ("X", "Y", "Z"):
        if geo in plot_df.columns:
            out[geo] = False
    if color_col in plot_df.columns and color_col not in out:
        out[color_col] = True
    if "Artists_Display" in plot_df.columns:
        out["Artists_Display"] = True
    return out


def scene_3d_layout() -> dict[str, Any]:
    """统一 3D 场景:暗色调,网格、背景与等比坐标。"""
    axis_common = dict(
        showbackground=True,
        backgroundcolor="rgb(34, 38, 49)",  # = --bg-card
        gridcolor="rgba(140, 148, 165, 0.22)",
        showgrid=True,
        zeroline=False,
        color="#8b8f99",  # 刻度文字
        title_font=dict(color="#d4d6db"),
    )
    return dict(
        xaxis=dict(title="PC1", **axis_common),
        yaxis=dict(title="PC2", **axis_common),
        zaxis=dict(title="PC3", **axis_common),
        aspectmode="data",
        camera=dict(eye=dict(x=1.45, y=1.45, z=1.15)),
        bgcolor="rgb(26, 29, 36)",  # = --bg-page
    )


def scatter_3d_discrete_groups(
    plot_df: pd.DataFrame,
    group_col: str,
    hover_numeric: list[str],
) -> go.Figure:
    """
    按类别(歌手 / 地区)拆成多条 3D 轨迹。一首歌只在其 Primary_Artist 类别中绘制一次,
    避免合作曲目在不同类别间重复出现造成视觉与统计偏差。

    特殊处理: 当 group_col == "Display_Region" 时, 使用固定的 C/J/K-pop 配色。
    """
    is_region_mode = group_col == "Display_Region"

    # 暗底友好的离散色板:中等饱和、中等亮度,避免粉糯/过亮
    palette = (
        px.colors.qualitative.T10
        + px.colors.qualitative.Plotly
        + px.colors.qualitative.G10
        + px.colors.qualitative.Vivid
    )

    # 用 astype(str) 把 Categorical 降级,避免把已选但 0 命中的类别也画一条空轨迹
    vc = plot_df[group_col].astype(str).value_counts()
    # 过滤掉 'nan' 等异常类别(若有)
    vc = vc[vc.index != "nan"]
    if is_region_mode:
        # 按 C → J → K → Other 的固定顺序排,而不是按数量降序; 三色三轴心智更稳
        region_order = [REGION_C, REGION_J, REGION_K, REGION_OTHER]
        cats = [r for r in region_order if r in vc.index]
    else:
        cats = vc.index.tolist()

    fig = go.Figure()
    extra_hover = [c for c in hover_numeric if c in plot_df.columns]

    for idx, cat in enumerate(cats):
        sub = plot_df[plot_df[group_col].astype(str) == str(cat)]
        if is_region_mode:
            rgb = get_region_color(str(cat))
        else:
            rgb = palette[idx % len(palette)]

        hover_lines = [
            "<b>%{text}</b>",
            "PC1 (X): %{x:.3f}",
            "PC2 (Y): %{y:.3f}",
            "PC3 (Z): %{z:.3f}",
        ]
        customdata_cols: list[np.ndarray] = []
        # First column is always the full artist credits (so hovering on a collab
        # makes both contributors immediately visible)
        if "Artists_Display" in sub.columns:
            customdata_cols.append(sub["Artists_Display"].astype(str).to_numpy())
            hover_lines.append("Artists: %{customdata[0]}")
        for j, cname in enumerate(extra_hover):
            customdata_cols.append(sub[cname].to_numpy())
            offset = 1 if "Artists_Display" in sub.columns else 0
            hover_lines.append(f"{cname}: %{{customdata[{j + offset}]:.4f}}")
        hover_lines.append("<extra></extra>")
        hovertemplate = "<br>".join(hover_lines)

        customdata = np.column_stack(customdata_cols) if customdata_cols else None

        track_text = (
            sub["Track_Name"].astype(str).to_numpy()
            if "Track_Name" in sub.columns
            else np.repeat("", len(sub))
        )

        n_songs = int(vc.loc[cat])
        label = str(cat)
        if len(label) > 40:
            label = label[:40] + "…"
        legend_name = f"{label} ({n_songs})"

        fig.add_trace(
            go.Scatter3d(
                x=sub["X"],
                y=sub["Y"],
                z=sub["Z"],
                mode="markers",
                name=legend_name,
                legendgroup=str(cat),
                text=track_text,
                customdata=customdata,
                hovertemplate=hovertemplate,
                marker=dict(
                    size=7,
                    color=rgb,
                    symbol="circle",
                    opacity=0.92,
                    line=dict(width=0.45, color="rgba(255,255,255,0.35)"),
                ),
            )
        )

    legend_title = (
        "Region (C/J/K-pop)" if is_region_mode else "Lead Artist (track count desc)"
    )
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=44),
        height=720,
        scene=scene_3d_layout(),
        legend=dict(
            title=dict(text=legend_title, font=dict(color="#d4d6db")),
            itemsizing="constant",
            tracegroupgap=0,
            font=dict(color="#d4d6db"),
            bgcolor="rgba(34, 38, 49, 0.6)",
            bordercolor="#353b48",
            borderwidth=1,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color="#d4d6db"),
    )
    return fig


# ==========================================
# 1. 页面全局配置
# ==========================================
st.set_page_config(
    page_title="East Asian Pop · Acoustic Feature Space",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================
# 全局样式:界面分区清晰、视觉层次柔和
# ==========================================
st.markdown(
    """
    <style>
    /* =========================================================
       暗色主题:三层灰阶 + 柔和文字色,避免纯黑纯白高对比
       变量:
         --bg-page:    页面底色(最深)
         --bg-card:    卡片底(中)
         --bg-soft:    Tab/侧栏容器底(略浅)
         --bg-hover:   hover 高亮底
         --border:     主分隔线
         --border-soft:更弱的分隔线
         --text:       主文字(柔和米灰,不用纯白)
         --text-muted: 副文字
         --accent:     强调色(低饱和暖灰蓝)
         --accent-soft:强调色对应的弱底
       ========================================================= */
    :root {
        --bg-page: #1a1d24;
        --bg-card: #222631;
        --bg-soft: #2a2f3a;
        --bg-hover: #323845;
        --border: #353b48;
        --border-soft: #2d323d;
        --text: #d4d6db;
        --text-muted: #8b8f99;
        --accent: #7c8aa3;
        --accent-soft: #2e3441;
    }

    /* 页面整体底色 */
    .stApp {
        background: var(--bg-page);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* 全局文字色调,Streamlit 默认会用纯白 */
    .stApp, .stApp p, .stApp li, .stApp label,
    .stApp .stMarkdown, .stMarkdown p {
        color: var(--text);
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: var(--text);
    }

    /* Hero:卡片底 + 左侧细色条 */
    .hero-header {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        padding: 1.4rem 1.8rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .hero-header h1 {
        color: var(--text) !important;
        margin: 0 0 0.45rem 0;
        font-size: 1.45rem;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    .hero-header p {
        color: var(--text-muted);
        margin: 0;
        font-size: 0.92rem;
        line-height: 1.65;
    }

    /* Tab 条 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--bg-soft);
        padding: 4px;
        border-radius: 10px;
        border: 1px solid var(--border-soft);
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        padding: 0 16px;
        background: transparent;
        border-radius: 7px;
        font-weight: 500;
        color: var(--text-muted);
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--bg-hover);
        color: var(--text);
    }
    .stTabs [aria-selected="true"] {
        background: var(--bg-card) !important;
        color: var(--text) !important;
        border: 1px solid var(--border);
    }

    /* 指标卡 */
    div[data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
    }
    div[data-testid="stMetricLabel"] {
        color: var(--text-muted);
        font-size: 0.78rem;
        font-weight: 500;
    }
    div[data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-size: 1.45rem;
        font-weight: 600;
    }

    /* 侧栏 */
    section[data-testid="stSidebar"] {
        background: var(--bg-card);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.4rem;
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {
        color: var(--text);
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-size: 0.92rem;
        font-weight: 600;
        margin-top: 1.1rem;
        margin-bottom: 0.5rem;
        letter-spacing: 0.01em;
    }
    section[data-testid="stSidebar"] hr {
        border-color: var(--border-soft);
        margin: 0.9rem 0;
    }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: var(--text-muted);
    }

    /* 输入框、下拉、multiselect:暗底 + 浅边 */
    .stTextInput input, .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div {
        background: var(--bg-soft) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
    }
    .stTextInput input::placeholder {
        color: var(--text-muted) !important;
    }

    /* multiselect 中已选的 tag */
    .stMultiSelect [data-baseweb="tag"] {
        background: var(--accent-soft) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
    }

    /* radio 选项的标签字 */
    .stRadio label, .stRadio div {
        color: var(--text) !important;
    }

    /* 已选歌手 chip */
    .artist-chip {
        display: inline-flex;
        align-items: center;
        background: var(--accent-soft);
        border: 1px solid var(--border);
        color: var(--text);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        margin: 2px 4px 2px 0;
        font-weight: 500;
    }
    .artist-chip-count {
        color: var(--text-muted);
        font-weight: 400;
        margin-left: 4px;
    }

    /* 按钮:暗底 + 边框,hover 时仅替换边色 */
    .stButton button {
        border-radius: 7px;
        font-weight: 500;
        background: var(--bg-soft);
        border: 1px solid var(--border);
        color: var(--text);
        transition: all 0.15s ease;
    }
    .stButton button:hover {
        background: var(--bg-hover);
        border-color: var(--accent);
        color: var(--text);
    }

    /* 主区 #### 子标题 */
    .block-container h4 {
        color: var(--text);
        font-weight: 600;
        font-size: 1rem;
        margin-top: 0.4rem;
        margin-bottom: 0.8rem;
    }

    /* 表格:暗底 */
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }

    /* 提示框(info / warning / error):降低饱和度 */
    div[data-testid="stAlert"] {
        background: var(--bg-soft);
        border: 1px solid var(--border);
        color: var(--text);
    }
    div[data-testid="stAlert"] * {
        color: var(--text) !important;
    }

    /* 分隔线 */
    hr {
        border-color: var(--border) !important;
    }

    /* caption / 小字 */
    .stApp [data-testid="stCaptionContainer"],
    .stApp small {
        color: var(--text-muted);
    }

    /* DataFrame 内部背景 */
    div[data-testid="stDataFrame"] [data-testid="stTable"],
    div[data-testid="stDataFrame"] table {
        background: var(--bg-card);
        color: var(--text);
    }

    /* multiselect / selectbox 下拉弹层 */
    div[data-baseweb="popover"],
    ul[role="listbox"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
    }
    li[role="option"] {
        color: var(--text) !important;
    }
    li[role="option"]:hover {
        background: var(--bg-hover) !important;
    }

    /* Plotly 容器:消除可能的白色描边 */
    .js-plotly-plot, .plotly, .plot-container {
        background: transparent !important;
    }

    /* radio 圆点:暗底 */
    .stRadio [role="radiogroup"] label {
        color: var(--text);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# Header
# ==========================================
st.markdown(
    """
    <div class="hero-header">
        <h1>🎵 East Asian Pop Music: PCA & Acoustic Feature Space</h1>
        <p>Acoustic parameters (Acousticness, Energy, Valence and more) are projected from
        a 9-dimensional feature space down to 3D via PCA. Points that sit close together
        share a similar sonic profile.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 数据加载
# ==========================================
@st.cache_data
def load_pca(csv_path_str: str) -> pd.DataFrame:
    return pd.read_csv(Path(csv_path_str), encoding="utf-8-sig")


@st.cache_data
def load_features(csv_path_str: str) -> Optional[pd.DataFrame]:
    path = Path(csv_path_str)
    if not path.is_file():
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data
def load_loadings(csv_path_str: str) -> Optional[pd.DataFrame]:
    """加载主成分载荷矩阵(行=原始声学特征,列=PC1/PC2/PC3)。"""
    path = Path(csv_path_str)
    if not path.is_file():
        return None
    try:
        ld = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
        return ld
    except Exception:
        return None


def merge_extra_features(
    pca_df: pd.DataFrame,
    feat_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    把完整特征表中独有的列(如 Danceability / Speechiness / Liveness / Loudness / Tempo)
    并入 PCA 表。雷达图与情绪象限需要这些列,而 PCA CSV 中只保留了 X/Y/Z 与三个核心指标。
    """
    if feat_df is None or feat_df.empty:
        return pca_df.copy()

    keys: list[str] = []
    if "Track_Name" in pca_df.columns and "Track_Name" in feat_df.columns:
        keys.append("Track_Name")

    # 试图把 features 表中的歌手列对齐到 pca 表中的歌手列名
    pca_artist_col = next(
        (c for c in ("Artist", "artists", "track_artist") if c in pca_df.columns),
        None,
    )
    feat_artist_col = next(
        (c for c in ("Artist", "artists", "track_artist") if c in feat_df.columns),
        None,
    )
    feat_use = feat_df.copy()
    if pca_artist_col and feat_artist_col:
        if feat_artist_col != pca_artist_col:
            feat_use = feat_use.rename(columns={feat_artist_col: pca_artist_col})
        keys.append(pca_artist_col)

    if not keys:
        return pca_df.copy()

    extra_cols = [c for c in feat_use.columns if c not in set(pca_df.columns)]
    if not extra_cols:
        return pca_df.copy()
    pick = keys + extra_cols
    try:
        merged = pca_df.merge(feat_use[pick], on=keys, how="left")
    except Exception:
        return pca_df.copy()
    return merged


required_xyz = {"X", "Y", "Z"}

try:
    df = load_pca(str(PCA_CSV))
except FileNotFoundError:
    st.error(
        f"Data file `{PCA_CSV.name}` not found. "
        "Place it next to `app.py`, then run: `streamlit run app.py`"
    )
    st.stop()

missing_xyz = sorted(required_xyz - set(df.columns))
if missing_xyz:
    st.error(
        f"Data file is missing required columns `{missing_xyz}`. "
        "Re-run `process_pca.py` to regenerate it."
    )
    st.stop()

artist_candidates = ["Artist", "artists", "track_artist"]
artist_col: Optional[str] = next((c for c in artist_candidates if c in df.columns), None)

if artist_col:
    df[artist_col] = df[artist_col].fillna("(Unknown Artist)")
    df = attach_artist_columns(df, artist_col)
    df["Artists_Display"] = df["Artists_List"].apply(
        lambda xs: ", ".join(xs) if xs else "(Unknown Artist)"
    )

df_features = load_features(str(FEATURES_CSV))
df_loadings = load_loadings(str(LOADINGS_CSV))
df = merge_extra_features(df, df_features)
numeric_hover = [c for c in ("Acousticness", "Energy", "Valence") if c in df.columns]


# ==========================================
# Sidebar: artist selection, region grouping, color settings
# ==========================================
st.sidebar.markdown("## 🎚️ Control Panel")

# Region visibility state (global; valid even when artist_col is missing)
# Each region can be independently toggled on/off in the charts.
REGION_VIS_KEY = "_region_visibility_v1"
if REGION_VIS_KEY not in st.session_state:
    st.session_state[REGION_VIS_KEY] = {
        REGION_C: True,
        REGION_J: True,
        REGION_K: True,
        REGION_OTHER: True,
    }

if artist_col:
    # Flat single-artist pool (deduped, sorted by total participation count desc)
    single_artist_pool = build_single_artist_pool(df)
    artist_play_count = count_per_single_artist(df)
    _canon_set_lower = {a.lower(): a for a in single_artist_pool}

    if "artist_search_input" not in st.session_state:
        st.session_state["artist_search_input"] = ""

    BASKET_KEY = "_artist_selected_basket_v2"
    if BASKET_KEY not in st.session_state:
        st.session_state[BASKET_KEY] = []

    # Defensive: drop artists no longer present in the data
    st.session_state[BASKET_KEY] = [
        x for x in st.session_state[BASKET_KEY] if x.lower() in _canon_set_lower
    ]
    basket_set_lower = {x.lower() for x in st.session_state[BASKET_KEY]}

    # ========== Quick-add by region ==========
    st.sidebar.markdown("### 🌏 Add Artists by Region")
    st.sidebar.caption(
        "One click adds **all artists** in that region to your basket. "
        "Toggle **Aggregate by region** below to collapse them into one "
        "averaged trace per region in every chart."
    )
    region_buckets = collect_artists_by_region(single_artist_pool)

    # Row 1: C / J / K
    region_btn_cols = st.sidebar.columns(3)
    region_btn_specs = [
        (REGION_C, "🇨🇳 C-pop", region_btn_cols[0]),
        (REGION_J, "🇯🇵 J-pop", region_btn_cols[1]),
        (REGION_K, "🇰🇷 K-pop", region_btn_cols[2]),
    ]
    for region, label, col in region_btn_specs:
        n_in_region = len(region_buckets[region])
        col.button(
            f"{label}\n({n_in_region})",
            key=f"add_region_{region}",
            disabled=(n_in_region == 0),
            use_container_width=True,
            help=f"Add all {n_in_region} {region} artists",
        )

    # Row 2: Others + Select all
    extra_btn_cols = st.sidebar.columns(2)
    n_other = len(region_buckets[REGION_OTHER])
    extra_btn_cols[0].button(
        f"🌐 Others\n({n_other})",
        key=f"add_region_{REGION_OTHER}",
        disabled=(n_other == 0),
        use_container_width=True,
        help=(
            f"Add all {n_other} artists not classified as C/J/K-pop. "
            "Useful for English-only collaborators or non-East-Asian acts in the dataset."
        ),
    )
    n_all = len(single_artist_pool)
    extra_btn_cols[1].button(
        f"⭐ Select all\n({n_all})",
        key="add_all_artists",
        disabled=(n_all == 0),
        use_container_width=True,
        help=f"Add every one of the {n_all} artists in the dataset to your basket.",
    )

    # Handle clicks: 4 region buttons + select-all (one rerun handles all)
    _click_handled = False
    for region, _label, _col in region_btn_specs + [
        (REGION_OTHER, None, None),
    ]:
        if st.session_state.get(f"add_region_{region}", False):
            cur_lower = {x.lower() for x in st.session_state[BASKET_KEY]}
            for a in region_buckets[region]:
                if a.lower() not in cur_lower:
                    st.session_state[BASKET_KEY].append(a)
                    cur_lower.add(a.lower())
            _click_handled = True
    if st.session_state.get("add_all_artists", False):
        cur_lower = {x.lower() for x in st.session_state[BASKET_KEY]}
        for a in single_artist_pool:
            if a.lower() not in cur_lower:
                st.session_state[BASKET_KEY].append(a)
                cur_lower.add(a.lower())
        _click_handled = True
    if _click_handled:
        st.session_state[BASKET_KEY] = sorted(
            st.session_state[BASKET_KEY],
            key=lambda a: -artist_play_count.get(a, 0),
        )
        st.rerun()

    # ========== Aggregate-by-region toggle ==========
    AGG_KEY = "_aggregate_by_region_v1"
    if AGG_KEY not in st.session_state:
        st.session_state[AGG_KEY] = False
    st.sidebar.toggle(
        "🎯 Aggregate by region",
        key=AGG_KEY,
        help=(
            "ON  : each region collapses into a single averaged 'virtual artist'. "
            "Radar shows up to 4 traces; 3D and the mood quadrant color points by region.\n"
            "OFF : individual-artist view (default)."
        ),
    )

    # ========== Region visibility (drop a region to compare a subset) ==========
    st.sidebar.markdown("**Show regions in charts**")
    st.sidebar.caption(
        "Uncheck any region to hide it. Lets you focus on, say, J-pop vs K-pop only."
    )
    vis_cols = st.sidebar.columns(4)
    for i, region in enumerate([REGION_C, REGION_J, REGION_K, REGION_OTHER]):
        # Each toggle binds to a fresh key, then we sync back into the dict
        toggle_key = f"_vis_toggle_{region}"
        if toggle_key not in st.session_state:
            st.session_state[toggle_key] = bool(
                st.session_state[REGION_VIS_KEY].get(region, True)
            )
        with vis_cols[i]:
            checked = st.checkbox(
                region,
                key=toggle_key,
                help=f"Show {region} in all charts",
            )
            st.session_state[REGION_VIS_KEY][region] = bool(checked)

    st.sidebar.divider()

    # ========== Search ==========
    st.sidebar.markdown("### 🔎 Search Artists")
    st.sidebar.text_input(
        "search",
        placeholder="Type an artist name, e.g. BTS / Jay Chou",
        key="artist_search_input",
        label_visibility="collapsed",
    )
    needle = str(st.session_state.get("artist_search_input") or "").strip().lower()

    if needle:
        pool = [
            a
            for a in single_artist_pool
            if needle in a.lower() and a.lower() not in basket_set_lower
        ]
        st.sidebar.caption(f"🎯 **{len(pool)}** artists match")
    else:
        pool = [a for a in single_artist_pool if a.lower() not in basket_set_lower]
        st.sidebar.caption(f"📋 **{len(pool)}** artists available to add")

    if pool:
        display_options = [
            f"{a}  ·  {artist_play_count.get(a, 0)} tracks" for a in pool
        ]
        display_to_name = dict(zip(display_options, pool))

        st.sidebar.multiselect(
            "candidates",
            options=display_options,
            key="candidate_multiselect_add",
            label_visibility="collapsed",
            placeholder="Pick from candidates...",
        )
        col_add, col_clear = st.sidebar.columns([1, 1])
        with col_add:
            if st.button("➕ Add", use_container_width=True):
                picks_disp = list(st.session_state.get("candidate_multiselect_add") or [])
                picks = [display_to_name[d] for d in picks_disp if d in display_to_name]
                cur_lower = {x.lower() for x in st.session_state[BASKET_KEY]}
                for p in picks:
                    if p.lower() not in cur_lower:
                        st.session_state[BASKET_KEY].append(p)
                        cur_lower.add(p.lower())
                st.session_state[BASKET_KEY] = sorted(
                    st.session_state[BASKET_KEY],
                    key=lambda a: -artist_play_count.get(a, 0),
                )
                if "candidate_multiselect_add" in st.session_state:
                    del st.session_state["candidate_multiselect_add"]
                st.rerun()
        with col_clear:
            if st.button("🗑️ Clear all", use_container_width=True):
                st.session_state[BASKET_KEY] = []
                st.rerun()
    else:
        st.sidebar.caption("(No more candidates)")

    st.sidebar.divider()

    # ========== Selected basket ==========
    st.sidebar.markdown("### ✓ Selected Artists")
    selected_ordered = sorted(
        st.session_state[BASKET_KEY],
        key=lambda a: -artist_play_count.get(a, 0),
    )

    if not selected_ordered:
        st.sidebar.info(
            "No artists selected yet. Use the region buttons or the search box above.",
            icon="💡",
        )
    else:
        st.sidebar.caption(f"**{len(selected_ordered)}** selected")
        # When the basket is huge (e.g. after Select all), don't render every chip;
        # show the first N and a brief summary, otherwise the sidebar becomes unscrollable
        MAX_CHIPS = 60
        for name in selected_ordered[:MAX_CHIPS]:
            col_a, col_b = st.sidebar.columns([5, 1.4])
            n_tracks = artist_play_count.get(name, 0)
            col_a.markdown(
                f"<span class='artist-chip'>{name}"
                f"<span class='artist-chip-count'>· {n_tracks}</span></span>",
                unsafe_allow_html=True,
            )
            _hid = hashlib.md5(str(name).encode("utf-8")).hexdigest()
            if col_b.button("✕", key=f"rm_{_hid}", help=f"Remove {name}"):
                st.session_state[BASKET_KEY] = [
                    x for x in st.session_state[BASKET_KEY] if x != name
                ]
                st.rerun()
        if len(selected_ordered) > MAX_CHIPS:
            st.sidebar.caption(
                f"… and **{len(selected_ordered) - MAX_CHIPS}** more "
                "(use Clear all to reset, or filter via the toggles above)"
            )

    selected_artists = selected_ordered

    # ========== "any-artist" inclusion filtering ==========
    if selected_artists:
        df_plot = filter_by_any_artist(df, selected_artists)
        # Lock Display_Artist as Categorical so:
        #   1) categories == selected_artists (legend never shows unselected artists)
        #   2) order == user's selection order (stable legend across re-renders)
        if "Display_Artist" in df_plot.columns and not df_plot.empty:
            df_plot["Display_Artist"] = pd.Categorical(
                df_plot["Display_Artist"],
                categories=selected_artists,
                ordered=True,
            )
        # Display_Region locked to C → J → K → Other order so colors stay aligned
        # even when a particular region has no surviving rows
        if "Display_Region" in df_plot.columns and not df_plot.empty:
            df_plot["Display_Region"] = pd.Categorical(
                df_plot["Display_Region"],
                categories=[REGION_C, REGION_J, REGION_K, REGION_OTHER],
                ordered=True,
            )
    else:
        df_plot = df.iloc[0:0].copy()
else:
    df_plot = df.copy()
    st.sidebar.warning(
        "No artist column in the data, so artist filtering is unavailable. "
        "Run `extract_kaggle_data.py` then `process_pca.py` first."
    )

# Aggregate-by-region flag (only meaningful when artist_col exists)
if artist_col:
    aggregate_by_region: bool = bool(st.session_state.get(AGG_KEY, False))
else:
    aggregate_by_region = False

# Apply region-visibility filter: drop rows whose Display_Region is unchecked.
# This must run BEFORE any chart consumes df_plot so every chart sees the same
# filtered data and they stay synchronized.
visible_regions: set[str] = {
    r for r, on in st.session_state[REGION_VIS_KEY].items() if on
}
if "Display_Region" in df_plot.columns and not df_plot.empty:
    df_plot = df_plot[df_plot["Display_Region"].astype(str).isin(visible_regions)].copy()

# ========== Color settings ==========
st.sidebar.divider()
st.sidebar.markdown("### 🎨 3D Scatter Coloring")

color_choices_meta: list[tuple[str, str]] = []
if artist_col and "Display_Region" in df_plot.columns and aggregate_by_region:
    # Aggregate mode: prefer "by region", hide "by individual artist"
    color_choices_meta.append(("🌏 By region", "Display_Region"))
elif artist_col and "Display_Artist" in df_plot.columns:
    color_choices_meta.append(("👤 By artist", "Display_Artist"))
    if "Display_Region" in df_plot.columns:
        color_choices_meta.append(("🌏 By region", "Display_Region"))
for label, key in (
    ("⚡ By Energy", "Energy"),
    ("😊 By Valence", "Valence"),
    ("🎻 By Acousticness", "Acousticness"),
):
    if key in df_plot.columns:
        color_choices_meta.append((label, key))

if not color_choices_meta:
    st.error("No usable column found for coloring. Please check the CSV.")
    st.stop()

radio_labels = [x[0] for x in color_choices_meta]
label_to_col = dict(color_choices_meta)
color_label = st.sidebar.radio(
    "color_method",
    radio_labels,
    index=0,
    label_visibility="collapsed",
)
color_target = label_to_col[color_label]


# ==========================================
# Main area: tabs
# ==========================================
tab_overview, tab_3d, tab_dist, tab_radar, tab_mood = st.tabs(
    [
        "📊  Overview",
        "🌐  3D PCA Space",
        "📈  Feature Distributions",
        "🎯  Acoustic Radar",
        "🎭  Mood Quadrant",
    ]
)

# ---------- Tab 1: Overview ----------
with tab_overview:
    st.markdown("#### Filter Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tracks", f"{len(df_plot):,}")
    if artist_col and not df_plot.empty:
        unique_singles = len({a for xs in df_plot["Artists_List"] for a in xs})
        c2.metric("Distinct Artists", f"{unique_singles:,}")
    else:
        c2.metric("Distinct Artists", "—")
    c3.metric(
        "Mean Energy",
        f"{df_plot['Energy'].mean():.3f}"
        if "Energy" in df_plot.columns and not df_plot.empty
        else "—",
    )
    c4.metric(
        "Mean Valence",
        f"{df_plot['Valence'].mean():.3f}"
        if "Valence" in df_plot.columns and not df_plot.empty
        else "—",
    )

    # Region breakdown (shown when data is non-empty and Display_Region exists)
    if not df_plot.empty and "Display_Region" in df_plot.columns:
        region_counts: dict[str, int] = (
            df_plot["Display_Region"].astype(str).value_counts().to_dict()
        )
        st.markdown("")
        rc1, rc2, rc3, rc4 = st.columns(4)
        for col, region in zip(
            (rc1, rc2, rc3, rc4),
            (REGION_C, REGION_J, REGION_K, REGION_OTHER),
        ):
            n = int(region_counts.get(region, 0))
            color = REGION_COLORS[region]
            label = f"🎵 {region} tracks"
            col.markdown(
                f"<div style='background:var(--bg-card); border:1px solid var(--border);"
                f" border-left:3px solid {color}; border-radius:8px; padding:0.7rem 1rem;'>"
                f"<div style='color:var(--text-muted); font-size:0.78rem; font-weight:500;'>{label}</div>"
                f"<div style='color:var(--text); font-size:1.4rem; font-weight:600; margin-top:2px;'>{n:,}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.markdown("#### Data Preview (first 50 rows of current filter)")
    if df_plot.empty:
        st.info(
            "No artists selected, or no rows match the current filter. "
            "Use the region buttons or the search box on the left to add artists.",
            icon="🔍",
        )
    else:
        preview_cols = [
            c
            for c in [
                "Track_Name",
                "Artists_Display",
                "Display_Artist",
                "Display_Region",
                "Primary_Artist",
                "Energy",
                "Valence",
                "Acousticness",
            ]
            if c in df_plot.columns
        ]
        rest = [c for c in df_plot.columns if c not in preview_cols and c != "Artists_List"]
        show_df = df_plot[preview_cols + rest].head(50).rename(
            columns={
                "Track_Name": "Track",
                "Artists_Display": "Artists (full)",
                "Display_Artist": "Bucketed to (selected)",
                "Display_Region": "Region",
                "Primary_Artist": "Lead Artist",
            }
        )
        st.dataframe(show_df, use_container_width=True, hide_index=True)

# ---------- Tab 2: 3D PCA ----------
with tab_3d:
    st.markdown("#### 3D Principal Component Space + Loading Heatmap")
    if aggregate_by_region:
        st.caption(
            "🖱️ Drag to rotate · scroll to zoom · hover for track details. "
            "**Region-aggregate mode**: every track is colored by its region — "
            f"<span style='color:{REGION_COLORS[REGION_C]}'>● C-pop</span> · "
            f"<span style='color:{REGION_COLORS[REGION_J]}'>● J-pop</span> · "
            f"<span style='color:{REGION_COLORS[REGION_K]}'>● K-pop</span> · "
            f"<span style='color:{REGION_COLORS[REGION_OTHER]}'>● Others</span>. "
            "The **loading heatmap** on the right shows which raw acoustic features "
            "drive each principal component.",
            unsafe_allow_html=True,
        )
    else:
        st.caption(
            "🖱️ Drag to rotate · scroll to zoom · hover for track details. "
            "The legend only shows **selected artists**: collaborations are bucketed to "
            "the first selected artist on the credit list, so unselected collaborators "
            "never sneak in. The **loading heatmap** on the right reveals what each "
            "PC axis is physically capturing."
        )

    col_3d, col_heatmap = st.columns([7, 3])

    with col_3d:
        if df_plot.empty:
            st.warning(
                "Current filter is empty. Add at least one artist on the left, "
                "or re-enable a region in 'Show regions in charts'.",
                icon="⚠️",
            )
        elif color_target not in df_plot.columns:
            st.error(f"Color column `{color_target}` not present in the current data.")
        elif is_numeric_series(df_plot[color_target]):
            hover_data = build_hover_data(df_plot, color_target, numeric_hover)
            scatter_kw: dict[str, Any] = dict(
                data_frame=df_plot,
                x="X",
                y="Y",
                z="Z",
                color=color_target,
                hover_data=hover_data if hover_data else None,
                opacity=0.85,
                color_continuous_scale=px.colors.sequential.Viridis,
            )
            if "Track_Name" in df_plot.columns:
                scatter_kw["hover_name"] = "Track_Name"

            fig = px.scatter_3d(**scatter_kw)
            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=20),
                height=620,
                scene=scene_3d_layout(),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, system-ui, sans-serif", color="#d4d6db"),
                coloraxis_colorbar=dict(
                    tickfont=dict(color="#8b8f99"),
                    title=dict(font=dict(color="#d4d6db")),
                    outlinewidth=0,
                ),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = scatter_3d_discrete_groups(df_plot, color_target, numeric_hover)
            # 在 7:3 列下,3D 高度收一点更平衡
            fig.update_layout(height=620, margin=dict(l=0, r=0, b=0, t=20))
            st.plotly_chart(fig, use_container_width=True)

    with col_heatmap:
        st.markdown("**🔍 PCA Loadings**")
        if df_loadings is None or df_loadings.empty:
            st.info(
                "`pca_loadings.csv` not found. Re-run the updated `process_pca.py` "
                "to generate the loading matrix.",
                icon="ℹ️",
            )
        else:
            # Adapts to whatever the column names happen to be (PC1/PC2/PC3 or others)
            ld = df_loadings.copy()
            if ld.shape[1] >= 3:
                col_labels = [f"{c}" for c in ld.columns[:3]]
                ld = ld.iloc[:, :3]
            else:
                col_labels = list(ld.columns)

            fig_heat = px.imshow(
                ld,
                labels=dict(x="Principal Component", y="Acoustic Feature", color="Loading"),
                x=col_labels,
                y=ld.index,
                color_continuous_scale="RdBu_r",
                aspect="auto",
                zmin=-max(abs(ld.values.min()), abs(ld.values.max())),
                zmax=max(abs(ld.values.min()), abs(ld.values.max())),
            )
            fig_heat.update_layout(
                margin=dict(l=0, r=0, b=0, t=10),
                height=620,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, system-ui, sans-serif", color="#d4d6db"),
                coloraxis_colorbar=dict(
                    tickfont=dict(color="#8b8f99"),
                    title=dict(font=dict(color="#d4d6db")),
                    outlinewidth=0,
                ),
            )
            fig_heat.update_xaxes(color="#8b8f99")
            fig_heat.update_yaxes(color="#8b8f99")
            st.plotly_chart(fig_heat, use_container_width=True)
            st.caption("🔴 positive · 🔵 negative · darker = larger weight")

# ---------- Tab 3: Distributions ----------
with tab_dist:
    st.markdown("#### Acoustic Feature Distributions")
    metrics_in_data = [
        m for m in ("Energy", "Valence", "Acousticness") if m in df_plot.columns
    ]
    if not metrics_in_data or df_plot.empty:
        st.info(
            "No data to plot under the current filter, or Energy / Valence / "
            "Acousticness columns are missing.",
            icon="📊",
        )
    else:
        hist_fig = make_histogram_row(df_plot, metrics_in_data)
        st.plotly_chart(hist_fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Extended Feature Table (`east_asian_pop_features.csv`)")
    if df_features is None or df_features.empty:
        st.caption("File not found — this section can be ignored.")
    else:
        st.dataframe(df_features.head(80), use_container_width=True, hide_index=True)

# ---------- Tab 4: Acoustic Radar ----------
with tab_radar:
    if aggregate_by_region:
        st.markdown("#### Region Average Acoustic Fingerprint")
        st.caption(
            "Aggregate mode: each region collapses into a single virtual "
            "'average artist' — the mean of all its tracks across 6 normalized "
            "acoustic features. Up to 4 traces, with fixed colors: "
            f"<span style='color:{REGION_COLORS[REGION_C]}'>● C-pop</span> · "
            f"<span style='color:{REGION_COLORS[REGION_J]}'>● J-pop</span> · "
            f"<span style='color:{REGION_COLORS[REGION_K]}'>● K-pop</span> · "
            f"<span style='color:{REGION_COLORS[REGION_OTHER]}'>● Others</span>. "
            "Uncheck a region in the sidebar to drop its trace.",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("#### Artist Acoustic Fingerprints")
        st.caption(
            "One trace per selected artist, averaged over 6 normalized [0, 1] "
            "acoustic features. Collaborations are bucketed to the first selected "
            "artist on the credit list, so unselected collaborators never sneak in."
        )

    radar_avail = [c for c in RADAR_FEATURES if c in df_plot.columns]
    radar_missing = [c for c in RADAR_FEATURES if c not in df_plot.columns]

    if df_plot.empty:
        st.warning(
            "Current filter is empty. Add at least one artist on the left, "
            "or re-enable a region in 'Show regions in charts'.",
            icon="⚠️",
        )
    elif len(radar_avail) < 3:
        st.info(
            f"Missing radar features in the current data: `{radar_missing}`. "
            "Make sure `east_asian_pop_features.csv` is in the same folder and "
            "`process_pca.py` has been re-run.",
            icon="ℹ️",
        )
    else:
        # ---------- Group means ----------
        if aggregate_by_region and "Display_Region" in df_plot.columns:
            # Aggregate mode: one virtual "average artist" per region
            # .astype(str) flattens the Categorical so groupby drops empty buckets
            grp_col = df_plot["Display_Region"].astype(str)
            df_radar = (
                df_plot.assign(_grp=grp_col)
                .groupby("_grp", dropna=True, observed=True)[radar_avail]
                .mean()
                .reset_index()
                .rename(columns={"_grp": "Group"})
            )
            # Stable C → J → K → Others ordering, regardless of frequency
            region_order = [REGION_C, REGION_J, REGION_K, REGION_OTHER]
            df_radar["_order"] = df_radar["Group"].apply(
                lambda r: region_order.index(r) if r in region_order else 99
            )
            df_radar = df_radar.sort_values("_order").drop(columns="_order").reset_index(drop=True)
            name_col = "Group"
            color_lookup = {r: get_region_color(r) for r in region_order}
        elif "Display_Artist" in df_plot.columns:
            # Individual mode: one trace per selected artist
            df_radar = (
                df_plot.groupby("Display_Artist", dropna=True, observed=True)[radar_avail]
                .mean()
                .reset_index()
            )
            name_col = "Display_Artist"
            color_lookup = None
        else:
            mean_row = df_plot[radar_avail].mean()
            df_radar = pd.DataFrame([mean_row.tolist()], columns=radar_avail)
            df_radar["_group"] = "All tracks"
            name_col = "_group"
            color_lookup = None

        # Dark-friendly discrete palette (used in individual mode)
        radar_palette = (
            px.colors.qualitative.T10
            + px.colors.qualitative.Plotly
            + px.colors.qualitative.G10
        )

        theta = radar_avail + [radar_avail[0]]
        fig_radar = go.Figure()
        for i, (_, row) in enumerate(df_radar.iterrows()):
            rvals = [float(row[c]) for c in radar_avail]
            label = str(row[name_col])
            if color_lookup is not None:
                color = color_lookup.get(label, REGION_COLORS[REGION_OTHER])
            else:
                color = radar_palette[i % len(radar_palette)]
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=rvals + [rvals[0]],
                    theta=theta,
                    fill="toself",
                    name=label,
                    line=dict(color=color, width=1.8 if aggregate_by_region else 1.6),
                    opacity=0.5 if aggregate_by_region else 0.55,
                )
            )

        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(34, 38, 49, 0.5)",
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    color="#8b8f99",
                    gridcolor="rgba(140, 148, 165, 0.22)",
                    tickfont=dict(color="#8b8f99", size=10),
                ),
                angularaxis=dict(
                    color="#d4d6db",
                    gridcolor="rgba(140, 148, 165, 0.22)",
                    tickfont=dict(color="#d4d6db", size=12),
                ),
            ),
            showlegend=True,
            legend=dict(
                title=dict(
                    text="Region" if aggregate_by_region else "",
                    font=dict(color="#d4d6db"),
                ),
                font=dict(color="#d4d6db"),
                bgcolor="rgba(34, 38, 49, 0.6)",
                bordercolor="#353b48",
                borderwidth=1,
            ),
            margin=dict(l=40, r=40, b=40, t=40),
            height=560,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, system-ui, sans-serif", color="#d4d6db"),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # In aggregate mode, surface the sample size behind each averaged trace
        if aggregate_by_region and "Display_Region" in df_plot.columns:
            region_stats: list[dict[str, Any]] = []
            for r in [REGION_C, REGION_J, REGION_K, REGION_OTHER]:
                sub = df_plot[df_plot["Display_Region"].astype(str) == r]
                if sub.empty:
                    continue
                n_tracks = len(sub)
                n_artists = sub["Display_Artist"].astype(str).nunique() \
                    if "Display_Artist" in sub.columns else 0
                region_stats.append(
                    {"Region": r, "Tracks": n_tracks, "Selected artists": n_artists}
                )
            if region_stats:
                st.caption("📌 Sample size behind each region's average")
                st.dataframe(
                    pd.DataFrame(region_stats),
                    use_container_width=True,
                    hide_index=True,
                )


# ---------- Tab 5: Mood Quadrant ----------
with tab_mood:
    if aggregate_by_region:
        st.markdown("#### Mood Quadrant: Valence × Energy (colored by region)")
        st.caption(
            "X axis: Valence (sad ↔ happy). Y axis: Energy (calm ↔ intense). "
            "0.5 reference lines split the plane into four quadrants. In aggregate mode, "
            "**each track is colored by its region**, with ★ marking each region's "
            "centroid (mean Valence, mean Energy). Bubble size still maps to "
            "**Loudness (dB)** (min-max normalized within the current filter)."
        )
    else:
        st.markdown("#### Mood Quadrant: Valence × Energy")
        st.caption(
            "X axis: Valence (sad ↔ happy). Y axis: Energy (calm ↔ intense). "
            "0.5 reference lines split the plane into four mood quadrants. Bubble size "
            "maps to **Loudness (dB)** — since dB is usually negative, it's min-max "
            "normalized within the current filter before being used as size; "
            "hover shows the raw value."
        )

    if df_plot.empty:
        st.warning(
            "Current filter is empty. Add at least one artist on the left, "
            "or re-enable a region in 'Show regions in charts'.",
            icon="⚠️",
        )
    elif "Valence" not in df_plot.columns or "Energy" not in df_plot.columns:
        st.info(
            "Valence or Energy column missing — cannot draw the mood quadrant.",
            icon="ℹ️",
        )
    else:
        mood_df = df_plot.copy()
        # Categorical columns make px.scatter list every category in the legend
        # (even empty ones). Casting to str makes plotly only show categories that
        # actually have data points.
        if "Display_Artist" in mood_df.columns:
            mood_df["Display_Artist"] = mood_df["Display_Artist"].astype(str)
        if "Display_Region" in mood_df.columns:
            mood_df["Display_Region"] = mood_df["Display_Region"].astype(str)

        scatter_kw_mood: dict[str, Any] = dict(
            data_frame=mood_df,
            x="Valence",
            y="Energy",
            opacity=0.78,
        )
        if "Track_Name" in mood_df.columns:
            scatter_kw_mood["hover_name"] = "Track_Name"

        # Color strategy: aggregate mode → fixed 4-color region palette, otherwise per-artist
        if aggregate_by_region and "Display_Region" in mood_df.columns:
            scatter_kw_mood["color"] = "Display_Region"
            scatter_kw_mood["color_discrete_map"] = {
                REGION_C: REGION_COLORS[REGION_C],
                REGION_J: REGION_COLORS[REGION_J],
                REGION_K: REGION_COLORS[REGION_K],
                REGION_OTHER: REGION_COLORS[REGION_OTHER],
            }
            # Lock legend ordering: C → J → K → Others
            scatter_kw_mood["category_orders"] = {
                "Display_Region": [REGION_C, REGION_J, REGION_K, REGION_OTHER]
            }
        else:
            scatter_kw_mood["color_discrete_sequence"] = (
                px.colors.qualitative.T10
                + px.colors.qualitative.Plotly
                + px.colors.qualitative.G10
            )
            if "Display_Artist" in mood_df.columns:
                scatter_kw_mood["color"] = "Display_Artist"

        # Loudness → bubble size: must be non-negative, min-max normalized
        hover_extra: dict[str, Any] = {}
        if "Loudness" in mood_df.columns:
            l = pd.to_numeric(mood_df["Loudness"], errors="coerce")
            if l.notna().any():
                lo, hi = float(l.min()), float(l.max())
                if hi > lo:
                    bubble = (l.fillna(l.median()) - lo) / (hi - lo)
                else:
                    bubble = pd.Series(1.0, index=mood_df.index)
                mood_df["_bubble"] = bubble.clip(lower=0.05)
                scatter_kw_mood["data_frame"] = mood_df
                scatter_kw_mood["size"] = "_bubble"
                scatter_kw_mood["size_max"] = 16
                hover_extra["Loudness"] = ":.2f"
                hover_extra["_bubble"] = False

        if "Artists_Display" in mood_df.columns:
            hover_extra["Artists_Display"] = True
        if hover_extra:
            scatter_kw_mood["hover_data"] = hover_extra

        fig_mood = px.scatter(**scatter_kw_mood)

        # Aggregate mode: overlay per-region centroid (★)
        if aggregate_by_region and "Display_Region" in mood_df.columns:
            for r in [REGION_C, REGION_J, REGION_K, REGION_OTHER]:
                sub = mood_df[mood_df["Display_Region"] == r]
                if sub.empty:
                    continue
                cx = float(sub["Valence"].mean())
                cy = float(sub["Energy"].mean())
                fig_mood.add_trace(
                    go.Scatter(
                        x=[cx],
                        y=[cy],
                        mode="markers+text",
                        marker=dict(
                            symbol="star",
                            size=22,
                            color=REGION_COLORS[r],
                            line=dict(color="rgba(255,255,255,0.85)", width=1.6),
                        ),
                        text=[f" {r} mean"],
                        textposition="top right",
                        textfont=dict(color="#d4d6db", size=11),
                        name=f"{r} centroid",
                        legendgroup=r,
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{r} centroid</b><br>"
                            "Valence: %{x:.3f}<br>"
                            "Energy: %{y:.3f}<extra></extra>"
                        ),
                    )
                )

        # Quadrant reference lines
        fig_mood.add_vline(
            x=0.5, line_width=1, line_dash="dash", line_color="rgba(140, 148, 165, 0.4)"
        )
        fig_mood.add_hline(
            y=0.5, line_width=1, line_dash="dash", line_color="rgba(140, 148, 165, 0.4)"
        )

        # Quadrant labels (subtle)
        annot_style = dict(
            showarrow=False,
            font=dict(color="#8b8f99", size=11),
            bgcolor="rgba(34, 38, 49, 0.5)",
            borderpad=4,
        )
        fig_mood.add_annotation(x=0.97, y=0.97, xref="x", yref="y",
                                text="Happy & Energetic",
                                xanchor="right", yanchor="top", **annot_style)
        fig_mood.add_annotation(x=0.03, y=0.97, xref="x", yref="y",
                                text="Tense / Angry",
                                xanchor="left", yanchor="top", **annot_style)
        fig_mood.add_annotation(x=0.97, y=0.03, xref="x", yref="y",
                                text="Peaceful / Relaxed",
                                xanchor="right", yanchor="bottom", **annot_style)
        fig_mood.add_annotation(x=0.03, y=0.03, xref="x", yref="y",
                                text="Sad / Melancholy",
                                xanchor="left", yanchor="bottom", **annot_style)

        fig_mood.update_layout(
            height=620,
            margin=dict(l=20, r=20, b=20, t=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(34, 38, 49, 0.4)",
            font=dict(family="Inter, system-ui, sans-serif", color="#d4d6db"),
            legend=dict(
                font=dict(color="#d4d6db"),
                bgcolor="rgba(34, 38, 49, 0.6)",
                bordercolor="#353b48",
                borderwidth=1,
            ),
            xaxis=dict(
                range=[0, 1],
                gridcolor="rgba(140, 148, 165, 0.15)",
                color="#8b8f99",
                title=dict(font=dict(color="#d4d6db")),
            ),
            yaxis=dict(
                range=[0, 1],
                gridcolor="rgba(140, 148, 165, 0.15)",
                color="#8b8f99",
                title=dict(font=dict(color="#d4d6db")),
            ),
        )
        st.plotly_chart(fig_mood, use_container_width=True)


st.divider()
st.caption(
    "✨ PCA dimensionality reduction · Streamlit + Plotly · "
    "Artist filtering uses inclusive semantics: a track is kept whenever any of "
    "its credited artists is in your selection."
)