import os
import re
from datetime import datetime, timedelta, timezone

import isodate
import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# =========================================================
# 0. App setup
# =========================================================

st.set_page_config(
    page_title="Finder Competitor Content Dashboard",
    layout="wide"
)

st.title("Finder Competitor Content Intelligence Dashboard")
st.caption(
    "Reusable prototype: collect YouTube data through API, upload raw platform exports, "
    "upload prepared competitor workbooks, analyse comments, and generate competitor comparison outputs."
)


# =========================================================
# 1. API key logic
# =========================================================

def get_secret_api_key():
    try:
        if "YOUTUBE_API_KEY" in st.secrets:
            return st.secrets["YOUTUBE_API_KEY"]
    except Exception:
        pass

    env_key = os.getenv("YOUTUBE_API_KEY")
    if env_key:
        return env_key

    return None


# =========================================================
# 2. General helper functions
# =========================================================

def safe_lower(text):
    return str(text).lower()


def parse_number(value):
    """
    Handles values like 725.1k, 1.20M, 16700, 16.7K.
    """
    if pd.isna(value):
        return 0

    text = str(value).strip().replace(",", "").replace(" ", "")

    try:
        return int(float(text))
    except Exception:
        pass

    text_lower = text.lower()

    try:
        if text_lower.endswith("k"):
            return int(float(text_lower.replace("k", "")) * 1000)
        if text_lower.endswith("m"):
            return int(float(text_lower.replace("m", "")) * 1_000_000)
    except Exception:
        return 0

    return 0


def parse_percent(value):
    """
    Handles 1.84%, 1.84, 0.0184.
    Returns percentage number, e.g. 1.84.
    """
    if pd.isna(value):
        return None

    text = str(value).strip().replace("%", "")

    try:
        num = float(text)
        if 0 < num < 1:
            return num * 100
        return num
    except Exception:
        return None


def read_uploaded_file(uploaded_file, sheet_name=0, header=0):
    if uploaded_file is None:
        return None

    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    return pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header)


def infer_competitor_from_filename(filename):
    name = filename.rsplit(".", 1)[0]
    name = name.replace("_", " ").replace("-", " ")

    remove_words = [
        "tiktok", "youtube", "instagram", "comments", "comment",
        "export", "data", "last 90 days", "last 365 day", "last 365 days",
        "top performing", "videos", "video", "social media analysis",
        "competitor benchmarking", "finder"
    ]

    lower_name = name.lower()
    for word in remove_words:
        lower_name = lower_name.replace(word, "")

    cleaned = " ".join(lower_name.split()).title()

    return cleaned if cleaned else "Uploaded Competitor"


def parse_youtube_accounts(text):
    """
    Expected format:
    Competitor Name | YouTube URL or handle
    """
    accounts = []

    for line in str(text).splitlines():
        line = line.strip()
        if not line:
            continue

        if "|" in line:
            name, url = line.split("|", 1)
            accounts.append({
                "competitor": name.strip(),
                "youtube_input": url.strip()
            })
        else:
            accounts.append({
                "competitor": line.strip(),
                "youtube_input": line.strip()
            })

    return accounts


# =========================================================
# 3. YouTube API functions
# =========================================================

def extract_channel_id_or_handle(channel_input: str):
    if not channel_input:
        return {"type": "empty", "value": None}

    text = channel_input.strip()

    if text.startswith("UC") and len(text) > 10:
        return {"type": "channel_id", "value": text}

    match_channel = re.search(r"youtube\.com/channel/([^/?]+)", text)
    if match_channel:
        return {"type": "channel_id", "value": match_channel.group(1)}

    match_handle = re.search(r"youtube\.com/@([^/?]+)", text)
    if match_handle:
        return {"type": "handle", "value": "@" + match_handle.group(1)}

    if text.startswith("@"):
        return {"type": "handle", "value": text}

    return {"type": "search", "value": text}


def resolve_youtube_channel_id(channel_input: str, api_key: str):
    parsed = extract_channel_id_or_handle(channel_input)

    if parsed["type"] == "empty":
        return None, "No YouTube channel input provided."

    if parsed["type"] == "channel_id":
        return parsed["value"], None

    query = parsed["value"]

    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "channel",
        "maxResults": 1,
        "key": api_key
    }

    response = requests.get(url, params=params, timeout=20)
    data = response.json()

    if response.status_code != 200:
        return None, data.get("error", {}).get("message", "YouTube API error.")

    items = data.get("items", [])
    if not items:
        return None, f"No YouTube channel found for: {channel_input}"

    return items[0]["snippet"]["channelId"], None


def fetch_youtube_recent_videos(channel_id: str, api_key: str, days: int = 90, max_results: int = 50):
    published_after = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    video_ids = []
    next_page_token = None

    while len(video_ids) < max_results:
        search_url = "https://www.googleapis.com/youtube/v3/search"
        search_params = {
            "part": "snippet",
            "channelId": channel_id,
            "type": "video",
            "order": "date",
            "publishedAfter": published_after,
            "maxResults": min(50, max_results - len(video_ids)),
            "key": api_key
        }

        if next_page_token:
            search_params["pageToken"] = next_page_token

        response = requests.get(search_url, params=search_params, timeout=20)
        data = response.json()

        if response.status_code != 200:
            raise RuntimeError(data.get("error", {}).get("message", "YouTube API error."))

        for item in data.get("items", []):
            video_ids.append(item["id"]["videoId"])

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    if not video_ids:
        return pd.DataFrame()

    rows = []

    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i + 50]

        videos_url = "https://www.googleapis.com/youtube/v3/videos"
        videos_params = {
            "part": "snippet,statistics,contentDetails",
            "id": ",".join(chunk),
            "key": api_key
        }

        response = requests.get(videos_url, params=videos_params, timeout=20)
        data = response.json()

        if response.status_code != 200:
            raise RuntimeError(data.get("error", {}).get("message", "YouTube API error."))

        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            content = item.get("contentDetails", {})

            video_id = item.get("id")
            duration_iso = content.get("duration", "PT0S")

            try:
                duration_seconds = int(isodate.parse_duration(duration_iso).total_seconds())
            except Exception:
                duration_seconds = None

            rows.append({
                "Platform": "YouTube",
                "Post URL / Thread": f"https://www.youtube.com/watch?v={video_id}",
                "Title / Description": snippet.get("title", ""),
                "Published Date": snippet.get("publishedAt", ""),
                "Duration Seconds": duration_seconds,
                "Views": int(stats.get("viewCount", 0)),
                "Likes": int(stats.get("likeCount", 0)) if "likeCount" in stats else 0,
                "Comments": int(stats.get("commentCount", 0)) if "commentCount" in stats else 0
            })

    return pd.DataFrame(rows)


def extract_video_id_from_url(url):
    text = str(url)

    match = re.search(r"watch\?v=([^&]+)", text)
    if match:
        return match.group(1)

    match = re.search(r"youtu\.be/([^?&]+)", text)
    if match:
        return match.group(1)

    return None


def fetch_youtube_comments_for_video(video_id, api_key, max_comments=20):
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        url = "https://www.googleapis.com/youtube/v3/commentThreads"
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(100, max_comments - len(comments)),
            "textFormat": "plainText",
            "order": "relevance",
            "key": api_key
        }

        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(url, params=params, timeout=20)
        data = response.json()

        if response.status_code != 200:
            return comments

        for item in data.get("items", []):
            try:
                top_comment = item["snippet"]["topLevelComment"]["snippet"]
            except KeyError:
                continue

            comments.append({
                "Video ID": video_id,
                "Comment": top_comment.get("textDisplay", ""),
                "Comment Likes": int(top_comment.get("likeCount", 0)),
                "Published": top_comment.get("publishedAt", ""),
                "Author": top_comment.get("authorDisplayName", "")
            })

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    return comments


def fetch_youtube_comments_for_benchmark(benchmark_df, api_key, max_comments_per_video=20):
    all_comments = []
    youtube_rows = benchmark_df[benchmark_df["Platform"] == "YouTube"].copy()

    for _, row in youtube_rows.iterrows():
        video_url = row["Post URL / Thread"]
        video_id = extract_video_id_from_url(video_url)

        if not video_id:
            continue

        video_comments = fetch_youtube_comments_for_video(
            video_id=video_id,
            api_key=api_key,
            max_comments=max_comments_per_video
        )

        for comment in video_comments:
            comment["Competitor"] = row.get("Competitor", "")
            comment["Platform"] = "YouTube"
            comment["Title"] = row["Title / Description"]
            comment["Url"] = video_url

        all_comments.extend(video_comments)

    if not all_comments:
        return pd.DataFrame()

    comments_df = pd.DataFrame(all_comments)
    comments_df = standardise_comment_upload(comments_df)

    return comments_df


# =========================================================
# 4. Classification functions
# =========================================================

def classify_topic(text):
    t = safe_lower(text)

    if any(k in t for k in ["cost of living", "inflation", "rent", "bills", "prices", "coffee"]):
        return "Cost of Living"
    if any(k in t for k in ["student", "university", "graduate", "overdraft"]):
        return "Student Finance"
    if any(k in t for k in ["monzo", "revolut", "starling", "chase", "bank", "current account"]):
        return "Banking"
    if any(k in t for k in ["saving", "savings", "cash isa", "isa", "interest", "aer", "frugal"]):
        return "Frugality / Saving"
    if any(k in t for k in ["credit card", "amex", "barclaycard", "avios"]):
        return "Credit Cards"
    if any(k in t for k in ["invest", "investing", "stocks", "shares", "trading 212", "etoro", "pension"]):
        return "Investing / Personal Finance"
    if any(k in t for k in ["wealth", "rich", "millionaire", "money habits", "financial habits"]):
        return "Wealth Building"
    if any(k in t for k in ["mindset", "psychology", "behaviour", "behavior"]):
        return "Money Mindset"
    if any(k in t for k in ["debt", "net worth"]):
        return "Debt / Net Worth"
    if any(k in t for k in ["salary", "income", "pay"]):
        return "Net Worth / Salary"
    if any(k in t for k in ["klarna", "clearpay", "paypal", "bnpl", "buy now pay later"]):
        return "BNPL / Payments"
    if any(k in t for k in ["deal", "promo", "bonus", "offer", "free money", "discount"]):
        return "Rewards / Deals"
    if any(k in t for k in ["podcast", "collab", "collaboration", "interview"]):
        return "Podcast / Collab"

    return "General Personal Finance"


def classify_format(platform, title, duration_seconds=None):
    t = safe_lower(title)
    platform = str(platform).lower()

    if platform == "tiktok":
        return "Short video"
    if platform == "instagram":
        if any(k in t for k in ["reel", "short"]):
            return "Reel"
        return "Post"

    if any(k in t for k in ["review", "is it worth"]):
        return "Product Review"
    if any(k in t for k in [" vs ", "compare", "comparison", "best "]):
        return "Comparison / Ranking"
    if any(k in t for k in ["explained", "what is", "how to", "guide"]):
        return "Explainer"
    if any(k in t for k in ["offer", "promo", "bonus", "deal"]):
        return "Offer-led"

    if duration_seconds is not None:
        if duration_seconds <= 60:
            return "Short video"
        if duration_seconds <= 180:
            return "Short / Under 3 mins"
        return "Long-form video"

    return "General Video"


def classify_er_tier(platform, er_percent):
    platform = str(platform).lower()

    if pd.isna(er_percent):
        return "Unknown"

    if platform == "youtube":
        if er_percent < 1.5:
            return "Low"
        elif er_percent < 2.5:
            return "Medium"
        elif er_percent < 3.5:
            return "High"
        else:
            return "Very High"

    if platform == "tiktok":
        if er_percent < 1.5:
            return "Low"
        elif er_percent < 2.0:
            return "Medium"
        elif er_percent < 3.0:
            return "High"
        else:
            return "Very High"

    if platform == "instagram":
        if er_percent < 0.3:
            return "Low"
        elif er_percent < 1.0:
            return "Medium"
        elif er_percent < 3.0:
            return "High"
        else:
            return "Very High"

    return "Unknown"


# =========================================================
# 5. Raw platform export standardisation
# =========================================================

def standardise_platform_export(df, competitor_name="Uploaded Competitor", fallback_platform="Uploaded"):
    """
    Handles raw TikTok / Instagram / other platform export files.
    One row = one post/video.
    """
    df = df.copy()

    rename_map = {
        "Competitor": "Competitor",
        "competitor": "Competitor",

        "Platform": "Platform",
        "platform": "Platform",

        "Video title": "Title / Description",
        "Title": "Title / Description",
        "Description": "Title / Description",
        "Caption": "Title / Description",
        "caption": "Title / Description",
        "Title / Description": "Title / Description",

        "Post URL": "Post URL / Thread",
        "Video link": "Post URL / Thread",
        "URL": "Post URL / Thread",
        "Url": "Post URL / Thread",
        "Post URL / Thread": "Post URL / Thread",

        "Total views": "Views",
        "Views": "Views",
        "View Count": "Views",
        "Play Count": "Views",
        "plays": "Views",

        "Total likes": "Likes",
        "Likes": "Likes",
        "Hearts": "Likes",
        "likes": "Likes",

        "Total comment": "Comments",
        "Total comments": "Comments",
        "Comments": "Comments",
        "Replies": "Comments",
        "comments": "Comments",

        "Post time": "Published Date",
        "Published Date": "Published Date",
        "Date": "Published Date",
        "Published": "Published Date",

        "Format": "Format",
        "ER (%)": "ER (%)",
        "ER Tier": "ER Tier",
        "Topic / Content Pillar": "Topic / Content Pillar"
    }

    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})

    required = ["Title / Description", "Views", "Likes", "Comments"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Raw platform export is missing required columns: {missing}. "
            f"Current columns: {list(df.columns)}"
        )

    if "Competitor" not in df.columns:
        df["Competitor"] = competitor_name
    else:
        df["Competitor"] = df["Competitor"].fillna("").replace("", competitor_name)

    if "Platform" not in df.columns:
        df["Platform"] = fallback_platform
    else:
        df["Platform"] = df["Platform"].fillna("").replace("", fallback_platform)

    if "Post URL / Thread" not in df.columns:
        df["Post URL / Thread"] = ""

    if "Published Date" not in df.columns:
        df["Published Date"] = ""

    for col in ["Views", "Likes", "Comments"]:
        df[col] = df[col].apply(parse_number)

    if "Format" not in df.columns:
        df["Format"] = df.apply(
            lambda row: classify_format(row.get("Platform"), row.get("Title / Description")),
            axis=1
        )

    if "ER (%)" not in df.columns:
        df["ER (%)"] = df.apply(
            lambda row: ((row["Likes"] + row["Comments"]) / row["Views"] * 100)
            if row["Views"] and row["Views"] > 0 else 0,
            axis=1
        )
    else:
        df["ER (%)"] = df["ER (%)"].apply(parse_percent)
        df["ER (%)"] = df["ER (%)"].fillna(
            df.apply(
                lambda row: ((row["Likes"] + row["Comments"]) / row["Views"] * 100)
                if row["Views"] and row["Views"] > 0 else 0,
                axis=1
            )
        )

    df["ER (%)"] = df["ER (%)"].round(2)

    if "ER Tier" not in df.columns:
        df["ER Tier"] = df.apply(
            lambda row: classify_er_tier(row["Platform"], row["ER (%)"]),
            axis=1
        )

    if "Topic / Content Pillar" not in df.columns:
        df["Topic / Content Pillar"] = df["Title / Description"].apply(classify_topic)

    return df[[
        "Competitor",
        "Platform",
        "Post URL / Thread",
        "Title / Description",
        "Published Date",
        "Format",
        "Views",
        "Likes",
        "Comments",
        "ER (%)",
        "ER Tier",
        "Topic / Content Pillar"
    ]]


def infer_platform_from_filename(filename):
    f = filename.lower()
    if "tiktok" in f:
        return "TikTok"
    if "instagram" in f or "ig" in f:
        return "Instagram"
    if "youtube" in f or "yt" in f:
        return "YouTube"
    return "Uploaded"


# =========================================================
# 6. Prepared competitor workbook reader
# =========================================================

def read_prepared_competitor_workbook(uploaded_file):
    """
    Reads a prepared multi-sheet competitor workbook.
    Each competitor is usually one sheet.
    The actual table header can start below the first row.
    """
    all_frames = []

    xls = pd.ExcelFile(uploaded_file)

    skip_keywords = [
        "posting behaviour",
        "audience",
        "funnels",
        "engagement summary",
        "summary",
        "notes",
        "readme"
    ]

    for sheet in xls.sheet_names:
        sheet_lower = str(sheet).lower()

        if any(k in sheet_lower for k in skip_keywords):
            continue

        raw = pd.read_excel(uploaded_file, sheet_name=sheet, header=None)

        header_row = None
        for i in range(len(raw)):
            row_values = raw.iloc[i].astype(str).str.strip().tolist()
            if "Platform" in row_values and "Post URL / Thread" in row_values:
                header_row = i
                break

        if header_row is None:
            continue

        df = pd.read_excel(uploaded_file, sheet_name=sheet, header=header_row)
        df = df.dropna(how="all")

        if "Platform" not in df.columns or "Title / Description" not in df.columns:
            continue

        df["Competitor"] = sheet

        if "Post URL / Thread" not in df.columns:
            df["Post URL / Thread"] = ""

        if "Published Date" not in df.columns:
            df["Published Date"] = ""

        if "Format" not in df.columns:
            df["Format"] = df.apply(
                lambda row: classify_format(row.get("Platform"), row.get("Title / Description")),
                axis=1
            )

        for col in ["Views", "Likes", "Comments"]:
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].apply(parse_number)

        if "ER (%)" not in df.columns:
            df["ER (%)"] = df.apply(
                lambda row: ((row["Likes"] + row["Comments"]) / row["Views"] * 100)
                if row["Views"] and row["Views"] > 0 else 0,
                axis=1
            )
        else:
            df["ER (%)"] = df["ER (%)"].apply(parse_percent)
            df["ER (%)"] = df["ER (%)"].fillna(
                df.apply(
                    lambda row: ((row["Likes"] + row["Comments"]) / row["Views"] * 100)
                    if row["Views"] and row["Views"] > 0 else 0,
                    axis=1
                )
            )

        df["ER (%)"] = df["ER (%)"].round(2)

        if "ER Tier" not in df.columns:
            df["ER Tier"] = df.apply(
                lambda row: classify_er_tier(row["Platform"], row["ER (%)"]),
                axis=1
            )

        if "Topic / Content Pillar" not in df.columns:
            df["Topic / Content Pillar"] = df["Title / Description"].apply(classify_topic)

        df = df[[
            "Competitor",
            "Platform",
            "Post URL / Thread",
            "Title / Description",
            "Published Date",
            "Format",
            "Views",
            "Likes",
            "Comments",
            "ER (%)",
            "ER Tier",
            "Topic / Content Pillar"
        ]]

        df = df[df["Platform"].notna()]
        df = df[df["Title / Description"].notna()]

        all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()

    return pd.concat(all_frames, ignore_index=True)


# =========================================================
# 7. Comment analysis functions
# =========================================================

def classify_comment_sentiment(comment):
    text = str(comment).lower()

    positive_words = [
        "thanks", "thank you", "helpful", "great", "good", "love",
        "useful", "amazing", "excellent", "interesting", "agree"
    ]

    negative_words = [
        "wrong", "bad", "hate", "terrible", "useless", "expensive",
        "scam", "confusing", "not true", "disagree", "poor"
    ]

    if any(word in text for word in negative_words):
        return "Negative"

    if any(word in text for word in positive_words):
        return "Positive"

    return "Neutral"


def classify_comment_theme(comment):
    text = str(comment).lower()

    if any(k in text for k in ["bank", "monzo", "revolut", "starling", "chase", "account"]):
        return "Banking Discussion"
    if any(k in text for k in ["saving", "savings", "isa", "interest", "aer"]):
        return "Savings Discussion"
    if any(k in text for k in ["credit card", "amex", "barclaycard", "avios"]):
        return "Credit Card Discussion"
    if any(k in text for k in ["invest", "stocks", "shares", "pension", "trading"]):
        return "Investing Discussion"
    if any(k in text for k in ["debt", "loan", "mortgage"]):
        return "Debt / Mortgage Discussion"
    if "?" in text or any(k in text for k in ["how", "what", "why", "should i", "can i"]):
        return "Question / Advice Seeking"

    return "General Reaction"


def classify_question_type(comment):
    text = str(comment).lower()

    if "?" not in text and not any(k in text for k in ["how", "what", "why", "should i", "can i"]):
        return "Not a question"

    if any(k in text for k in ["bank", "account", "monzo", "revolut", "starling"]):
        return "Banking question"
    if any(k in text for k in ["saving", "isa", "interest"]):
        return "Savings question"
    if any(k in text for k in ["invest", "stocks", "shares", "pension"]):
        return "Investing question"
    if any(k in text for k in ["credit card", "amex", "barclaycard"]):
        return "Credit card question"

    return "General question"


def classify_comment_depth(comment):
    text = str(comment).strip()
    word_count = len(text.split())

    if word_count <= 5:
        return "Short reaction"
    if word_count <= 20:
        return "Moderate"

    return "Substantive / detailed"


def standardise_comment_upload(df):
    df = df.copy()

    rename_map = {
        "Competitor": "Competitor",
        "competitor": "Competitor",

        "Platform": "Platform",
        "platform": "Platform",

        "Video ID": "Video ID",
        "Video id": "Video ID",
        "video_id": "Video ID",

        "Comment": "Comment",
        "comment": "Comment",
        "Text": "Comment",
        "text": "Comment",

        "Comment Likes": "Comment Likes",
        "Likes": "Comment Likes",
        "likes": "Comment Likes",

        "Published": "Published",
        "Date": "Published",
        "Published Date": "Published",
        "published_at": "Published",

        "Author": "Author",
        "Username": "Author",
        "User": "Author",
        "author": "Author",

        "Title": "Title",
        "Video Title": "Title",

        "Url": "Url",
        "URL": "Url",
        "Video URL": "Url",

        "Sentiment": "Sentiment",
        "Theme": "Theme",
        "Question": "Question Type",
        "Question Type": "Question Type",
        "Comment Depth": "Comment Depth",
        "Depth": "Comment Depth"
    }

    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})

    if "Comment" not in df.columns:
        raise ValueError(
            f"Comment file is missing required column: Comment. Current columns: {list(df.columns)}"
        )

    default_cols = {
        "Competitor": "",
        "Platform": "",
        "Video ID": "",
        "Comment Likes": 0,
        "Published": "",
        "Author": "",
        "Title": "",
        "Url": "",
        "Sentiment": "",
        "Theme": "",
        "Question Type": "",
        "Comment Depth": ""
    }

    for col, default in default_cols.items():
        if col not in df.columns:
            df[col] = default

    df["Comment Likes"] = pd.to_numeric(df["Comment Likes"], errors="coerce").fillna(0).astype(int)

    df["Sentiment"] = df.apply(
        lambda row: row["Sentiment"] if str(row["Sentiment"]).strip() else classify_comment_sentiment(row["Comment"]),
        axis=1
    )

    df["Theme"] = df.apply(
        lambda row: row["Theme"] if str(row["Theme"]).strip() else classify_comment_theme(row["Comment"]),
        axis=1
    )

    df["Question Type"] = df.apply(
        lambda row: row["Question Type"] if str(row["Question Type"]).strip() else classify_question_type(row["Comment"]),
        axis=1
    )

    df["Comment Depth"] = df.apply(
        lambda row: row["Comment Depth"] if str(row["Comment Depth"]).strip() else classify_comment_depth(row["Comment"]),
        axis=1
    )

    return df[[
        "Competitor",
        "Platform",
        "Video ID",
        "Comment",
        "Comment Likes",
        "Published",
        "Author",
        "Title",
        "Url",
        "Sentiment",
        "Theme",
        "Question Type",
        "Comment Depth"
    ]]


def generate_comment_summary(comment_df):
    total_comments = len(comment_df)

    if total_comments == 0:
        return pd.DataFrame({
            "Dimension": ["Sentiment", "Comment depth", "Main themes", "Questions", "Community behaviour"],
            "Result": ["No comments", "No comments", "No themes", "No questions", "No discussion"]
        })

    sentiment_counts = comment_df["Sentiment"].value_counts(normalize=True)
    depth_counts = comment_df["Comment Depth"].value_counts(normalize=True)
    theme_counts = comment_df["Theme"].value_counts()

    top_sentiment = sentiment_counts.idxmax()
    if top_sentiment == "Neutral" and sentiment_counts.max() >= 0.5:
        sentiment_result = "Mostly neutral"
    elif top_sentiment == "Positive":
        sentiment_result = "Mostly positive"
    elif top_sentiment == "Negative":
        sentiment_result = "Mostly negative"
    else:
        sentiment_result = "Mixed sentiment"

    top_depth = depth_counts.idxmax()
    if top_depth == "Short reaction":
        depth_result = "Mostly short reactions"
    elif top_depth == "Moderate":
        depth_result = "Mostly moderate comments"
    else:
        depth_result = "More substantive / detailed comments"

    top_themes = theme_counts.head(3).index.tolist()
    theme_result = ", ".join(top_themes)

    non_question_share = (comment_df["Question Type"] == "Not a question").mean()
    if non_question_share >= 0.8:
        question_result = "Very limited"
    elif non_question_share >= 0.5:
        question_result = "Some questions"
    else:
        question_result = "Question-led discussion"

    substantive_share = (comment_df["Comment Depth"] == "Substantive / detailed").mean()
    if substantive_share < 0.15 and non_question_share >= 0.7:
        community_result = "Weak peer-to-peer discussion"
    elif substantive_share < 0.3:
        community_result = "Limited peer-to-peer discussion"
    else:
        community_result = "Moderate peer-to-peer discussion"

    return pd.DataFrame({
        "Dimension": [
            "Sentiment",
            "Comment depth",
            "Main themes",
            "Questions",
            "Community behaviour"
        ],
        "Result": [
            sentiment_result,
            depth_result,
            theme_result,
            question_result,
            community_result
        ]
    })


# =========================================================
# 8. Sidebar inputs
# =========================================================

st.sidebar.header("YouTube API Collection")

youtube_accounts_text = st.sidebar.text_area(
    "YouTube competitors, one per line",
    value=(
        "Nischa | https://www.youtube.com/@nischa\n"
        "Damien Talks Money | https://www.youtube.com/@DamienTalksMoney"
    ),
    help="Format: Competitor Name | YouTube Channel URL or Handle"
)

default_key = get_secret_api_key()

manual_api_key = st.sidebar.text_input(
    "YouTube API Key",
    value=default_key if default_key else "",
    type="password",
    help="Paste your YouTube Data API key here. On Streamlit Cloud, store this in Secrets instead."
)

days = st.sidebar.slider(
    "Time Window",
    min_value=30,
    max_value=180,
    value=90,
    step=30
)

max_videos = st.sidebar.slider(
    "Max YouTube videos per competitor",
    min_value=10,
    max_value=100,
    value=50,
    step=10
)

st.sidebar.markdown("---")
st.sidebar.header("Upload Port 1: Raw Platform Exports")

raw_platform_uploads = st.sidebar.file_uploader(
    "Upload raw TikTok / Instagram / other platform exports",
    type=["csv", "xlsx"],
    accept_multiple_files=True,
    help=(
        "One row = one post/video. Required columns: title/caption, views, likes, comments. "
        "File name should include competitor and platform if possible, e.g. Nischa_TikTok.xlsx."
    )
)

st.sidebar.markdown("---")
st.sidebar.header("Upload Port 2: Prepared Competitor Workbook")

prepared_workbook_upload = st.sidebar.file_uploader(
    "Upload prepared competitor benchmark workbook",
    type=["xlsx"],
    help=(
        "Use this for the workbook with competitor sheets such as Monzo, Nischa, Damien. "
        "Each sheet should contain a table with Platform, Post URL / Thread, Title / Description, Views, Likes, Comments."
    )
)

st.sidebar.markdown("---")
st.sidebar.header("Upload Port 3: Comment Files")

fetch_youtube_comments = st.sidebar.checkbox(
    "Fetch YouTube comments automatically",
    value=True
)

max_comments_per_video = st.sidebar.slider(
    "Max comments per YouTube video",
    min_value=5,
    max_value=100,
    value=20,
    step=5
)

comment_uploads = st.sidebar.file_uploader(
    "Upload TikTok / Instagram / other comment files",
    type=["csv", "xlsx"],
    accept_multiple_files=True,
    help="Required column: Comment. Optional: Competitor, Platform, Sentiment, Theme, Question Type, Comment Depth."
)

run_button = st.sidebar.button("Run Analysis")


# =========================================================
# 9. Tabs
# =========================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Benchmark Table",
    "2. Performance Summary",
    "3. Topic Analysis",
    "4. Comment Analysis",
    "5. Competitor Comparison",
    "6. Sharing / Deployment Notes"
])


# =========================================================
# 10. Run analysis
# =========================================================

if run_button:
    all_data = []
    api_key = manual_api_key.strip() if manual_api_key else None

    # -----------------------------
    # YouTube API collection
    # -----------------------------
    youtube_accounts = parse_youtube_accounts(youtube_accounts_text)

    if youtube_accounts and api_key:
        for account in youtube_accounts:
            comp_name = account["competitor"]
            yt_input = account["youtube_input"]

            try:
                with st.spinner(f"Resolving YouTube channel for {comp_name}..."):
                    channel_id, error = resolve_youtube_channel_id(yt_input, api_key)

                if error:
                    st.warning(f"{comp_name}: {error}")
                    continue

                with st.spinner(f"Fetching recent YouTube videos for {comp_name}..."):
                    yt_df = fetch_youtube_recent_videos(
                        channel_id=channel_id,
                        api_key=api_key,
                        days=days,
                        max_results=max_videos
                    )

                if yt_df.empty:
                    st.info(f"No YouTube videos found for {comp_name} in the selected time window.")
                else:
                    yt_df["Competitor"] = comp_name
                    yt_df = standardise_platform_export(
                        yt_df,
                        competitor_name=comp_name,
                        fallback_platform="YouTube"
                    )
                    all_data.append(yt_df)

            except Exception as e:
                st.error(f"YouTube collection failed for {comp_name}: {e}")

    elif youtube_accounts and not api_key:
        st.warning("Please provide a YouTube API key to fetch YouTube data.")

    # -----------------------------
    # Upload Port 1: raw platform exports
    # -----------------------------
    if raw_platform_uploads:
        for uploaded_file in raw_platform_uploads:
            try:
                raw_df = read_uploaded_file(uploaded_file)
                comp_name = infer_competitor_from_filename(uploaded_file.name)
                platform = infer_platform_from_filename(uploaded_file.name)

                platform_df = standardise_platform_export(
                    raw_df,
                    competitor_name=comp_name,
                    fallback_platform=platform
                )

                all_data.append(platform_df)

            except Exception as e:
                st.error(f"Raw platform export upload failed for {uploaded_file.name}: {e}")

    # -----------------------------
    # Upload Port 2: prepared competitor workbook
    # -----------------------------
    if prepared_workbook_upload is not None:
        try:
            prepared_df = read_prepared_competitor_workbook(prepared_workbook_upload)

            if prepared_df.empty:
                st.warning("Prepared competitor workbook was uploaded, but no valid benchmark tables were found.")
            else:
                all_data.append(prepared_df)

        except Exception as e:
            st.error(f"Prepared competitor workbook upload failed: {e}")

    # -----------------------------
    # Combine post/video data
    # -----------------------------
    if all_data:
        benchmark = pd.concat(all_data, ignore_index=True, sort=False)
        benchmark = benchmark.dropna(subset=["Title / Description"], how="any")
        st.session_state["benchmark"] = benchmark

        if fetch_youtube_comments and api_key:
            with st.spinner("Fetching YouTube comments..."):
                yt_comments_df = fetch_youtube_comments_for_benchmark(
                    benchmark_df=benchmark,
                    api_key=api_key,
                    max_comments_per_video=max_comments_per_video
                )

            if not yt_comments_df.empty:
                st.session_state["comments_df"] = yt_comments_df
                st.session_state["comment_summary"] = generate_comment_summary(yt_comments_df)
            else:
                st.info("No YouTube comments collected. Comments may be disabled or unavailable.")

    else:
        st.warning("No post/video data collected. Add YouTube API key, upload raw platform exports, or upload a prepared workbook.")

    # -----------------------------
    # Upload Port 3: comment files
    # -----------------------------
    if comment_uploads:
        uploaded_comment_frames = []

        for uploaded_file in comment_uploads:
            try:
                raw_comments = read_uploaded_file(uploaded_file)
                uploaded_comments_df = standardise_comment_upload(raw_comments)

                inferred_competitor = infer_competitor_from_filename(uploaded_file.name)
                inferred_platform = infer_platform_from_filename(uploaded_file.name)

                uploaded_comments_df["Competitor"] = uploaded_comments_df["Competitor"].fillna("").replace("", inferred_competitor)
                uploaded_comments_df["Platform"] = uploaded_comments_df["Platform"].fillna("").replace("", inferred_platform)

                uploaded_comment_frames.append(uploaded_comments_df)

            except Exception as e:
                st.error(f"Comment file upload failed for {uploaded_file.name}: {e}")

        if uploaded_comment_frames:
            uploaded_comments_all = pd.concat(uploaded_comment_frames, ignore_index=True)

            if "comments_df" in st.session_state:
                combined_comments = pd.concat(
                    [st.session_state["comments_df"], uploaded_comments_all],
                    ignore_index=True
                )
            else:
                combined_comments = uploaded_comments_all

            st.session_state["comments_df"] = combined_comments
            st.session_state["comment_summary"] = generate_comment_summary(combined_comments)


benchmark = st.session_state.get("benchmark", None)
comments_df = st.session_state.get("comments_df", None)
comment_summary = st.session_state.get("comment_summary", None)


# =========================================================
# 11. Render tab 1: Benchmark Tables
# =========================================================

with tab1:
    st.subheader("Standardised Benchmark Tables by Competitor and Platform")

    if benchmark is not None:
        for (competitor, platform), group_df in benchmark.groupby(["Competitor", "Platform"]):
            st.markdown(f"### {competitor} — {platform}")
            display_df = group_df.reset_index(drop=True)
            st.dataframe(display_df, width="stretch")

            csv = display_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"Download {competitor}_{platform}_benchmark.csv",
                data=csv,
                file_name=f"{competitor}_{platform}_benchmark.csv".replace(" ", "_"),
                mime="text/csv",
                key=f"download_{competitor}_{platform}"
            )

        st.markdown("---")
        st.markdown("### Full Combined Benchmark Table")
        st.dataframe(benchmark.reset_index(drop=True), width="stretch")

        full_csv = benchmark.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download full combined benchmark table as CSV",
            data=full_csv,
            file_name="full_combined_benchmark_table.csv",
            mime="text/csv"
        )

    else:
        st.info("Run analysis to generate benchmark tables.")


# =========================================================
# 12. Render tab 2: Performance Summary
# =========================================================

with tab2:
    st.subheader("Performance Summary")

    if benchmark is not None:
        col1, col2, col3, col4 = st.columns(4)

        total_posts = len(benchmark)
        avg_views = benchmark["Views"].mean()
        avg_er = benchmark["ER (%)"].mean()
        avg_comments = benchmark["Comments"].mean()

        col1.metric("Total Posts", f"{total_posts:,.0f}")
        col2.metric("Avg Views", f"{avg_views:,.0f}")
        col3.metric("Avg ER", f"{avg_er:.2f}%")
        col4.metric("Avg Comments", f"{avg_comments:.1f}")

        platform_summary = benchmark.groupby(["Competitor", "Platform"]).agg(
            Posts=("Title / Description", "count"),
            Avg_Views=("Views", "mean"),
            Avg_ER=("ER (%)", "mean"),
            Avg_Comments=("Comments", "mean")
        ).reset_index()

        platform_summary["Avg_Views"] = platform_summary["Avg_Views"].round(0)
        platform_summary["Avg_ER"] = platform_summary["Avg_ER"].round(2)
        platform_summary["Avg_Comments"] = platform_summary["Avg_Comments"].round(1)

        st.markdown("### Competitor + Platform Summary")
        st.dataframe(platform_summary, width="stretch")

        col_a, col_b = st.columns(2)

        with col_a:
            fig_views = px.bar(
                platform_summary,
                x="Competitor",
                y="Avg_Views",
                color="Platform",
                barmode="group",
                title="Average Views by Competitor and Platform"
            )
            st.plotly_chart(fig_views, width="stretch")

        with col_b:
            fig_er = px.bar(
                platform_summary,
                x="Competitor",
                y="Avg_ER",
                color="Platform",
                barmode="group",
                title="Average Engagement Rate by Competitor and Platform"
            )
            st.plotly_chart(fig_er, width="stretch")

        st.markdown("### Top Posts by Views")
        st.dataframe(
            benchmark.sort_values("Views", ascending=False).head(20),
            width="stretch"
        )

    else:
        st.info("Run analysis to see performance summary.")


# =========================================================
# 13. Render tab 3: Topic Analysis
# =========================================================

with tab3:
    st.subheader("Topic / Content Pillar Analysis")

    if benchmark is not None:
        topic_summary = benchmark.groupby(["Competitor", "Platform", "Topic / Content Pillar"]).agg(
            Posts=("Title / Description", "count"),
            Avg_Views=("Views", "mean"),
            Avg_ER=("ER (%)", "mean"),
            Avg_Comments=("Comments", "mean")
        ).reset_index()

        topic_summary["Avg_Views"] = topic_summary["Avg_Views"].round(0)
        topic_summary["Avg_ER"] = topic_summary["Avg_ER"].round(2)
        topic_summary["Avg_Comments"] = topic_summary["Avg_Comments"].round(1)

        st.dataframe(topic_summary, width="stretch")

        col_a, col_b = st.columns(2)

        with col_a:
            fig_topic_views = px.bar(
                topic_summary.sort_values("Avg_Views", ascending=False),
                x="Topic / Content Pillar",
                y="Avg_Views",
                color="Competitor",
                title="Average Views by Topic"
            )
            st.plotly_chart(fig_topic_views, width="stretch")

        with col_b:
            fig_topic_er = px.bar(
                topic_summary.sort_values("Avg_ER", ascending=False),
                x="Topic / Content Pillar",
                y="Avg_ER",
                color="Competitor",
                title="Average ER by Topic"
            )
            st.plotly_chart(fig_topic_er, width="stretch")

        st.markdown("### Format Summary")

        format_summary = benchmark.groupby(["Competitor", "Platform", "Format"]).agg(
            Posts=("Title / Description", "count"),
            Avg_Views=("Views", "mean"),
            Avg_ER=("ER (%)", "mean"),
            Avg_Comments=("Comments", "mean")
        ).reset_index()

        format_summary["Avg_Views"] = format_summary["Avg_Views"].round(0)
        format_summary["Avg_ER"] = format_summary["Avg_ER"].round(2)
        format_summary["Avg_Comments"] = format_summary["Avg_Comments"].round(1)

        st.dataframe(format_summary, width="stretch")

    else:
        st.info("Run analysis to see topic analysis.")


# =========================================================
# 14. Render tab 4: Comment Analysis
# =========================================================

with tab4:
    st.subheader("Comment Analysis")

    if comments_df is not None:
        st.markdown("### Comment-Level Coding Table")
        st.dataframe(comments_df, width="stretch")

        st.markdown("### Overall Comment Summary")
        st.dataframe(comment_summary, width="stretch")

        st.markdown("### Comment Summary by Competitor and Platform")

        comment_group_summary = comments_df.groupby(["Competitor", "Platform"]).agg(
            Comments_Analysed=("Comment", "count"),
            Avg_Comment_Likes=("Comment Likes", "mean"),
            Substantive_Share=(
                "Comment Depth",
                lambda x: (x == "Substantive / detailed").mean()
            ),
            Question_Share=(
                "Question Type",
                lambda x: (x != "Not a question").mean()
            )
        ).reset_index()

        comment_group_summary["Avg_Comment_Likes"] = comment_group_summary["Avg_Comment_Likes"].round(1)
        comment_group_summary["Substantive_Share"] = (comment_group_summary["Substantive_Share"] * 100).round(1)
        comment_group_summary["Question_Share"] = (comment_group_summary["Question_Share"] * 100).round(1)

        st.dataframe(comment_group_summary, width="stretch")

        csv_comments = comments_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download comment-level table as CSV",
            data=csv_comments,
            file_name="comment_level_analysis.csv",
            mime="text/csv"
        )

        csv_summary = comment_summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download comment summary as CSV",
            data=csv_summary,
            file_name="comment_summary.csv",
            mime="text/csv"
        )

        st.markdown("### Comment Distribution")

        col1, col2 = st.columns(2)

        with col1:
            sentiment_chart = comments_df["Sentiment"].value_counts().reset_index()
            sentiment_chart.columns = ["Sentiment", "Count"]

            fig_sentiment = px.bar(
                sentiment_chart,
                x="Sentiment",
                y="Count",
                title="Comment Sentiment Distribution"
            )
            st.plotly_chart(fig_sentiment, width="stretch")

        with col2:
            depth_chart = comments_df["Comment Depth"].value_counts().reset_index()
            depth_chart.columns = ["Comment Depth", "Count"]

            fig_depth = px.bar(
                depth_chart,
                x="Comment Depth",
                y="Count",
                title="Comment Depth Distribution"
            )
            st.plotly_chart(fig_depth, width="stretch")

        theme_chart = comments_df["Theme"].value_counts().reset_index()
        theme_chart.columns = ["Theme", "Count"]

        fig_theme = px.bar(
            theme_chart,
            x="Theme",
            y="Count",
            title="Main Comment Themes"
        )
        st.plotly_chart(fig_theme, width="stretch")

    else:
        st.info(
            "Run YouTube analysis with automatic comment fetching, "
            "or upload TikTok / Instagram / other comment files."
        )


# =========================================================
# 15. Render tab 5: Competitor Comparison Matrix
# =========================================================

with tab5:
    st.subheader("Competitor Comparison Matrix")

    if benchmark is not None:
        comparison = benchmark.groupby(["Competitor", "Platform"]).agg(
            Posts=("Title / Description", "count"),
            Avg_Views=("Views", "mean"),
            Median_Views=("Views", "median"),
            Total_Views=("Views", "sum"),
            Avg_ER=("ER (%)", "mean"),
            Avg_Comments=("Comments", "mean"),
            Total_Comments=("Comments", "sum")
        ).reset_index()

        comparison["Avg_Views"] = comparison["Avg_Views"].round(0)
        comparison["Median_Views"] = comparison["Median_Views"].round(0)
        comparison["Avg_ER"] = comparison["Avg_ER"].round(2)
        comparison["Avg_Comments"] = comparison["Avg_Comments"].round(1)

        topic_perf = benchmark.groupby(
            ["Competitor", "Platform", "Topic / Content Pillar"]
        ).agg(
            Topic_Posts=("Title / Description", "count"),
            Topic_Avg_ER=("ER (%)", "mean"),
            Topic_Avg_Views=("Views", "mean")
        ).reset_index()

        topic_perf = topic_perf.sort_values(
            ["Competitor", "Platform", "Topic_Avg_ER"],
            ascending=[True, True, False]
        )

        top_topic = topic_perf.groupby(["Competitor", "Platform"]).head(1)[
            ["Competitor", "Platform", "Topic / Content Pillar"]
        ].rename(columns={"Topic / Content Pillar": "Top Topic by ER"})

        comparison = comparison.merge(
            top_topic,
            on=["Competitor", "Platform"],
            how="left"
        )

        format_counts = benchmark.groupby(
            ["Competitor", "Platform", "Format"]
        ).size().reset_index(name="Format_Count")

        format_counts = format_counts.sort_values(
            ["Competitor", "Platform", "Format_Count"],
            ascending=[True, True, False]
        )

        top_format = format_counts.groupby(["Competitor", "Platform"]).head(1)[
            ["Competitor", "Platform", "Format"]
        ].rename(columns={"Format": "Dominant Format"})

        comparison = comparison.merge(
            top_format,
            on=["Competitor", "Platform"],
            how="left"
        )

        if comments_df is not None and "Competitor" in comments_df.columns:
            comment_comp = comments_df.groupby(["Competitor", "Platform"]).agg(
                Comments_Analysed=("Comment", "count"),
                Avg_Comment_Likes=("Comment Likes", "mean"),
                Substantive_Share=(
                    "Comment Depth",
                    lambda x: (x == "Substantive / detailed").mean()
                ),
                Question_Share=(
                    "Question Type",
                    lambda x: (x != "Not a question").mean()
                )
            ).reset_index()

            comment_comp["Avg_Comment_Likes"] = comment_comp["Avg_Comment_Likes"].round(1)
            comment_comp["Substantive_Share"] = (comment_comp["Substantive_Share"] * 100).round(1)
            comment_comp["Question_Share"] = (comment_comp["Question_Share"] * 100).round(1)

            comparison = comparison.merge(
                comment_comp,
                on=["Competitor", "Platform"],
                how="left"
            )

        if "Finder" in comparison["Competitor"].values:
            finder_base = comparison[comparison["Competitor"] == "Finder"][
                ["Platform", "Avg_Views", "Avg_ER", "Avg_Comments"]
            ].rename(columns={
                "Avg_Views": "Finder_Avg_Views",
                "Avg_ER": "Finder_Avg_ER",
                "Avg_Comments": "Finder_Avg_Comments"
            })

            comparison = comparison.merge(finder_base, on="Platform", how="left")

            def compare_metric(value, finder_value):
                if pd.isna(finder_value) or finder_value == 0:
                    return "No Finder baseline"

                if value > finder_value * 1.2:
                    return "Better than Finder"
                elif value < finder_value * 0.8:
                    return "Weaker than Finder"
                else:
                    return "Similar to Finder"

            comparison["Reach vs Finder"] = comparison.apply(
                lambda row: compare_metric(row["Avg_Views"], row.get("Finder_Avg_Views")),
                axis=1
            )

            comparison["ER vs Finder"] = comparison.apply(
                lambda row: compare_metric(row["Avg_ER"], row.get("Finder_Avg_ER")),
                axis=1
            )

            comparison["Comments vs Finder"] = comparison.apply(
                lambda row: compare_metric(row["Avg_Comments"], row.get("Finder_Avg_Comments")),
                axis=1
            )

            comparison = comparison.drop(
                columns=["Finder_Avg_Views", "Finder_Avg_ER", "Finder_Avg_Comments"],
                errors="ignore"
            )

        st.dataframe(comparison, width="stretch")

        comparison_csv = comparison.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download competitor comparison table as CSV",
            data=comparison_csv,
            file_name="competitor_comparison_matrix.csv",
            mime="text/csv"
        )

    else:
        st.info("Run analysis to generate competitor comparison matrix.")


# =========================================================
# 16. Render tab 6: Sharing / Deployment Notes
# =========================================================

with tab6:
    st.subheader("Sharing / Deployment Notes")

    st.markdown(
        """
        ### Upload Port Design

        **Upload Port 1: Raw Platform Exports**  
        Use this for single-platform exports, such as TikTok or Instagram post-level data.  
        Required fields: title/caption, views, likes, comments.

        **Upload Port 2: Prepared Competitor Benchmark Workbook**  
        Use this for a prepared workbook where each competitor has a sheet, such as Monzo, Nischa or Damien.  
        The table header should contain: Platform, Post URL / Thread, Title / Description, Views, Likes, Comments.

        **Upload Port 3: Comment Files**  
        Use this for TikTok / Instagram / other platform comment files.  
        Required field: Comment.  
        Optional fields: Competitor, Platform, Sentiment, Theme, Question Type, Comment Depth.

        ### Deployment

        Required files:
        - `app.py`
        - `requirements.txt`

        Suggested `requirements.txt`:
        ```
        streamlit
        pandas
        requests
        plotly
        isodate
        openpyxl
        ```

        To deploy online:
        1. Upload `app.py` and `requirements.txt` to GitHub.
        2. Deploy through Streamlit Community Cloud.
        3. Add your YouTube key in Streamlit Secrets:
        ```
        YOUTUBE_API_KEY = "your_api_key_here"
        ```

        Do not put your API key directly into the code before uploading to GitHub.
        """
    )