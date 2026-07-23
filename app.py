import os
import re
from datetime import datetime, timedelta, timezone

import isodate
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from openai import OpenAI

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =========================================================
# 0. App setup
# =========================================================

st.set_page_config(
    page_title="Finder Competitor Content Dashboard",
    layout="wide"
)

st.title("Finder Competitor Content Intelligence Dashboard")
st.caption(
    "Dashboard data → AI competitor summary → AI marketing recommendations → ML content predictor"
)


# =========================================================
# 1. API key helpers
# =========================================================

def get_youtube_api_key():
    try:
        if "YOUTUBE_API_KEY" in st.secrets:
            return st.secrets["YOUTUBE_API_KEY"]
    except Exception:
        pass

    return os.getenv("YOUTUBE_API_KEY")


def get_openai_api_key():
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

    return os.getenv("OPENAI_API_KEY")


# =========================================================
# 2. General helpers
# =========================================================

def parse_number(value):
    if pd.isna(value):
        return 0

    text = str(value).strip().replace(",", "").replace(" ", "")

    try:
        return int(float(text))
    except Exception:
        pass

    try:
        if text.lower().endswith("k"):
            return int(float(text[:-1]) * 1000)
        if text.lower().endswith("m"):
            return int(float(text[:-1]) * 1_000_000)
    except Exception:
        return 0

    return 0


def parse_percent(value):
    if pd.isna(value):
        return None

    try:
        num = float(str(value).replace("%", "").strip())
        if 0 < num < 1:
            return num * 100
        return num
    except Exception:
        return None


def extract_hashtags(text):
    """
    Extract author-provided hashtags from video title and description.
    Example:
    'Watch this #investing #howtoinvest' -> '#investing, #howtoinvest'
    """
    if pd.isna(text):
        return ""

    hashtags = re.findall(r"#\w+", str(text))
    hashtags = list(dict.fromkeys(hashtags))

    return ", ".join(hashtags)


def read_uploaded_file(uploaded_file, sheet_name=0, header=0):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)

    return pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header)


def infer_competitor_from_filename(filename):
    name = filename.rsplit(".", 1)[0]
    name = name.replace("_", " ").replace("-", " ")

    remove_words = [
        "tiktok", "youtube", "instagram", "comments", "comment",
        "export", "data", "finder", "videos", "video",
        "benchmark", "analysis", "social media", "competitor",
        "top performing", "last 365 days", "last 365 day", "last 90 days"
    ]

    clean = name.lower()

    for word in remove_words:
        clean = clean.replace(word, "")

    clean = " ".join(clean.split()).title()

    return clean if clean else "Uploaded Competitor"


def infer_platform_from_filename(filename):
    file = filename.lower()

    if "tiktok" in file:
        return "TikTok"
    if "instagram" in file or "ig" in file:
        return "Instagram"
    if "youtube" in file or "yt" in file:
        return "YouTube"

    return "Uploaded"


def parse_youtube_accounts(text):
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


def render_competitor_platform_filters(df, key_prefix):
    col1, col2, col3 = st.columns(3)

    competitors = sorted(df["Competitor"].dropna().astype(str).unique().tolist())
    platforms = sorted(df["Platform"].dropna().astype(str).unique().tolist())

    with col1:
        selected_competitors = st.multiselect(
            "Filter competitors",
            options=competitors,
            default=competitors,
            key=f"{key_prefix}_competitors"
        )

    with col2:
        selected_platforms = st.multiselect(
            "Filter platforms",
            options=platforms,
            default=platforms,
            key=f"{key_prefix}_platforms"
        )

    with col3:
        sort_mode = st.radio(
            "Display order",
            options=["By platform", "By competitor"],
            horizontal=True,
            key=f"{key_prefix}_sort_mode"
        )

    filtered = df[
        df["Competitor"].astype(str).isin(selected_competitors)
        & df["Platform"].astype(str).isin(selected_platforms)
    ].copy()

    return filtered, sort_mode


# =========================================================
# 3. YouTube API functions
# =========================================================

def extract_channel_id_or_handle(channel_input):
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


def resolve_youtube_channel_id(channel_input, api_key):
    parsed = extract_channel_id_or_handle(channel_input)

    if parsed["type"] == "empty":
        return None, "No YouTube channel input provided."

    if parsed["type"] == "channel_id":
        return parsed["value"], None

    if parsed["type"] == "handle":
        handle = parsed["value"]

        url = "https://www.googleapis.com/youtube/v3/channels"
        params = {
            "part": "snippet,statistics,contentDetails",
            "forHandle": handle,
            "key": api_key
        }

        response = requests.get(url, params=params, timeout=20)
        data = response.json()

        if response.status_code == 200 and data.get("items"):
            return data["items"][0]["id"], None

        query = handle.replace("@", "")

    else:
        query = parsed["value"]

    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "channel",
        "maxResults": 5,
        "key": api_key
    }

    response = requests.get(url, params=params, timeout=20)
    data = response.json()

    if response.status_code != 200:
        return None, data.get("error", {}).get("message", "YouTube API error.")

    items = data.get("items", [])

    if not items:
        return None, f"No YouTube channel found for: {channel_input}"

    query_clean = query.lower().replace("@", "").replace(" ", "")
    best_item = items[0]

    for item in items:
        title = item.get("snippet", {}).get("channelTitle", "")
        title_clean = title.lower().replace(" ", "")

        if query_clean in title_clean or title_clean in query_clean:
            best_item = item
            break

    return best_item["snippet"]["channelId"], None


def fetch_youtube_channel_subscriber_count(channel_id, api_key):
    url = "https://www.googleapis.com/youtube/v3/channels"
    params = {
        "part": "statistics",
        "id": channel_id,
        "key": api_key
    }

    response = requests.get(url, params=params, timeout=20)
    data = response.json()

    if response.status_code != 200:
        return None

    items = data.get("items", [])

    if not items:
        return None

    stats = items[0].get("statistics", {})

    if stats.get("hiddenSubscriberCount") is True:
        return None

    subscriber_count = stats.get("subscriberCount")

    if subscriber_count is None:
        return None

    try:
        return int(subscriber_count)
    except Exception:
        return None


def fetch_youtube_recent_videos(
    channel_id,
    api_key,
    days=180,
    include_shorts=False,
    max_pages=10,
    subscriber_count=None
):
    """
    Fetch all recent YouTube videos within the selected time window.
    Hashtags are extracted from author-provided title + description.
    """

    published_after = (
        datetime.now(timezone.utc) - timedelta(days=days)
    ).isoformat()

    video_ids = []
    next_page_token = None
    page_count = 0

    while True:
        if page_count >= max_pages:
            break

        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "channelId": channel_id,
            "type": "video",
            "order": "date",
            "publishedAfter": published_after,
            "maxResults": 50,
            "key": api_key
        }

        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(url, params=params, timeout=20)
        data = response.json()

        if response.status_code != 200:
            raise RuntimeError(
                data.get("error", {}).get("message", "YouTube API error.")
            )

        for item in data.get("items", []):
            video_id = item.get("id", {}).get("videoId")
            if video_id:
                video_ids.append(video_id)

        next_page_token = data.get("nextPageToken")
        page_count += 1

        if not next_page_token:
            break

    if not video_ids:
        return pd.DataFrame()

    rows = []

    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i + 50]

        url = "https://www.googleapis.com/youtube/v3/videos"
        params = {
            "part": "snippet,statistics,contentDetails",
            "id": ",".join(chunk),
            "key": api_key
        }

        response = requests.get(url, params=params, timeout=20)
        data = response.json()

        if response.status_code != 200:
            raise RuntimeError(
                data.get("error", {}).get("message", "YouTube API error.")
            )

        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            content = item.get("contentDetails", {})

            video_id = item.get("id")
            duration_iso = content.get("duration", "PT0S")

            try:
                duration_seconds = int(
                    isodate.parse_duration(duration_iso).total_seconds()
                )
            except Exception:
                duration_seconds = 0

            if not include_shorts and duration_seconds <= 60:
                continue

            title = snippet.get("title", "")
            description = snippet.get("description", "")
            combined_text = f"{title} {description}"

            rows.append({
                "Platform": "YouTube",
                "Post URL / Thread": f"https://www.youtube.com/watch?v={video_id}",
                "Title / Description": title,
                "Published Date": snippet.get("publishedAt", ""),
                "Duration Seconds": duration_seconds,
                "Views": int(stats.get("viewCount", 0)),
                "Likes": int(stats.get("likeCount", 0)) if "likeCount" in stats else 0,
                "Comments": int(stats.get("commentCount", 0)) if "commentCount" in stats else 0,
                "Hashtags": extract_hashtags(combined_text),
                "Follower / Subscriber Count": subscriber_count,
                "Date Collected": datetime.now().date().isoformat()
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
                comment = item["snippet"]["topLevelComment"]["snippet"]
            except KeyError:
                continue

            comments.append({
                "Video ID": video_id,
                "Comment": comment.get("textDisplay", ""),
                "Comment Likes": int(comment.get("likeCount", 0)),
                "Published": comment.get("publishedAt", ""),
                "Author": comment.get("authorDisplayName", "")
            })

        next_page_token = data.get("nextPageToken")

        if not next_page_token:
            break

    return comments


def fetch_youtube_comments_for_benchmark(benchmark_df, api_key, max_comments_per_video):
    all_comments = []

    youtube_rows = benchmark_df[benchmark_df["Platform"] == "YouTube"].copy()

    for _, row in youtube_rows.iterrows():
        video_id = extract_video_id_from_url(row["Post URL / Thread"])

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
            comment["Url"] = row["Post URL / Thread"]

        all_comments.extend(video_comments)

    if not all_comments:
        return pd.DataFrame()

    return standardise_comment_upload(pd.DataFrame(all_comments))


# =========================================================
# 4. Classification functions
# =========================================================

def classify_topic(text):
    t = str(text).lower()

    if any(k in t for k in ["pension", "retirement"]):
        return "AD — Pension"
    if any(k in t for k in ["cost of living", "inflation", "rent", "bills", "prices"]):
        return "Cost of Living"
    if any(k in t for k in ["student", "university", "graduate", "overdraft"]):
        return "Student Finance"
    if any(k in t for k in ["monzo", "revolut", "starling", "chase", "bank", "current account"]):
        return "Banking"
    if any(k in t for k in ["saving", "savings", "cash isa", "isa", "interest", "frugal"]):
        return "Frugality / Saving"
    if any(k in t for k in ["credit card", "amex", "barclaycard", "avios"]):
        return "Credit Cards"
    if any(k in t for k in ["invest", "investing", "stocks", "shares", "trading", "portfolio"]):
        return "Investing / Personal Finance"
    if any(k in t for k in ["wealth", "rich", "millionaire", "money habits"]):
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
    if any(k in t for k in ["podcast", "collab", "collaboration", "interview", "double take"]):
        return "Podcast / Collab"

    return "General Personal Finance"


def classify_format(platform, title, duration_seconds=None):
    t = str(title).lower()
    p = str(platform).lower()

    if p == "tiktok":
        return "Short video"

    if p == "instagram":
        if any(k in t for k in ["reel", "short"]):
            return "Reel"
        return "Post"

    if any(k in t for k in ["shorts", "#shorts"]):
        return "Short video"
    if any(k in t for k in ["review", "is it worth"]):
        return "Product Review"
    if any(k in t for k in [" vs ", "compare", "comparison", "best "]):
        return "Comparison / Ranking"
    if any(k in t for k in ["explained", "what is", "how to", "guide"]):
        return "Explainer"
    if any(k in t for k in ["offer", "promo", "bonus", "deal"]):
        return "Offer-led"

    try:
        if duration_seconds is not None and not pd.isna(duration_seconds):
            duration_seconds = float(duration_seconds)

            if duration_seconds <= 60:
                return "Short video"
            if duration_seconds <= 180:
                return "Short / Under 3 mins"
            if duration_seconds <= 600:
                return "Medium / 3–10 mins"
            return "Long-form video"
    except Exception:
        pass

    return "General Video"


def classify_er_tier(platform, er):
    if pd.isna(er):
        return "Unknown"

    p = str(platform).lower()

    if p == "youtube":
        if er < 1.5:
            return "Low"
        if er < 2.5:
            return "Medium"
        if er < 3.5:
            return "High"
        return "Very High"

    if p == "tiktok":
        if er < 1.5:
            return "Low"
        if er < 2.0:
            return "Medium"
        if er < 3.0:
            return "High"
        return "Very High"

    if p == "instagram":
        if er < 0.3:
            return "Low"
        if er < 1.0:
            return "Medium"
        if er < 3.0:
            return "High"
        return "Very High"

    return "Unknown"


# =========================================================
# 5. Data standardisation
# =========================================================

def standardise_platform_export(df, competitor_name, fallback_platform):
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

        "Content": "Post URL / Thread",
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

        "Hashtags": "Hashtags",
        "hashtags": "Hashtags",
        "Hash Tags": "Hashtags",
        "Tags": "Hashtags",

        "Follower / Subscriber Count": "Follower / Subscriber Count",
        "Followers": "Follower / Subscriber Count",
        "Follower Count": "Follower / Subscriber Count",
        "Subscriber Count": "Follower / Subscriber Count",
        "Subscribers": "Follower / Subscriber Count",

        "Date Collected": "Date Collected",
        "Collection Date": "Date Collected",
        "Collected Date": "Date Collected",

        "Post time": "Published Date",
        "Video publish time": "Published Date",
        "Published Date": "Published Date",
        "Date": "Published Date",
        "Published": "Published Date",

        "Duration": "Duration Seconds",
        "Duration Seconds": "Duration Seconds",

        "Format": "Format",
        "ER (%)": "ER (%)",
        "Engagement Rate": "ER (%)",
        "ER Tier": "ER Tier",
        "Topic / Content Pillar": "Topic / Content Pillar",
        "Topic": "Topic / Content Pillar",
        "Content Pillar": "Topic / Content Pillar"
    }

    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})

    if "Likes" not in df.columns:
        df["Likes"] = 0

    if "Comments" not in df.columns:
        df["Comments"] = 0

    if "Hashtags" not in df.columns:
        if "Title / Description" in df.columns:
            df["Hashtags"] = df["Title / Description"].apply(extract_hashtags)
        else:
            df["Hashtags"] = ""

    if "Follower / Subscriber Count" not in df.columns:
        df["Follower / Subscriber Count"] = None

    if "Date Collected" not in df.columns:
        df["Date Collected"] = datetime.now().date().isoformat()

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

    if "Duration Seconds" not in df.columns:
        df["Duration Seconds"] = None

    for col in [
        "Views",
        "Likes",
        "Comments",
        "Duration Seconds",
        "Follower / Subscriber Count"
    ]:
        if col in df.columns:
            df[col] = df[col].apply(parse_number)

    if "Format" not in df.columns:
        df["Format"] = df.apply(
            lambda row: classify_format(
                row.get("Platform"),
                row.get("Title / Description"),
                row.get("Duration Seconds")
            ),
            axis=1
        )

    if "ER (%)" not in df.columns:
        df["ER (%)"] = df.apply(
            lambda row: (
                (row["Likes"] + row["Comments"]) / row["Views"] * 100
            )
            if row["Views"] and row["Views"] > 0 else 0,
            axis=1
        )
    else:
        df["ER (%)"] = df["ER (%)"].apply(parse_percent)

        calculated_er = df.apply(
            lambda row: (
                (row["Likes"] + row["Comments"]) / row["Views"] * 100
            )
            if row["Views"] and row["Views"] > 0 else 0,
            axis=1
        )

        df["ER (%)"] = df["ER (%)"].fillna(calculated_er)

    df["ER (%)"] = df["ER (%)"].round(2)

    if "ER Tier" not in df.columns:
        df["ER Tier"] = df.apply(
            lambda row: classify_er_tier(row["Platform"], row["ER (%)"]),
            axis=1
        )

    if "Topic / Content Pillar" not in df.columns:
        df["Topic / Content Pillar"] = df["Title / Description"].apply(classify_topic)

    return df[
        [
            "Competitor",
            "Platform",
            "Post URL / Thread",
            "Title / Description",
            "Published Date",
            "Duration Seconds",
            "Format",
            "Views",
            "Likes",
            "Comments",
            "Hashtags",
            "Follower / Subscriber Count",
            "Date Collected",
            "ER (%)",
            "ER Tier",
            "Topic / Content Pillar"
        ]
    ]


def read_prepared_competitor_workbook(uploaded_file):
    all_frames = []
    xls = pd.ExcelFile(uploaded_file)

    skip_keywords = [
        "posting behaviour", "audience", "funnels",
        "engagement summary", "summary", "notes", "readme"
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

        prepared = standardise_platform_export(
            df,
            competitor_name=sheet,
            fallback_platform="Uploaded"
        )

        all_frames.append(prepared)

    if not all_frames:
        return pd.DataFrame()

    return pd.concat(all_frames, ignore_index=True)


# =========================================================
# 6. Comment analysis
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
            f"Comment file is missing required column: Comment. "
            f"Current columns: {list(df.columns)}"
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

    df["Comment Likes"] = pd.to_numeric(
        df["Comment Likes"],
        errors="coerce"
    ).fillna(0).astype(int)

    df["Sentiment"] = df.apply(
        lambda row: row["Sentiment"]
        if str(row["Sentiment"]).strip()
        else classify_comment_sentiment(row["Comment"]),
        axis=1
    )

    df["Theme"] = df.apply(
        lambda row: row["Theme"]
        if str(row["Theme"]).strip()
        else classify_comment_theme(row["Comment"]),
        axis=1
    )

    df["Question Type"] = df.apply(
        lambda row: row["Question Type"]
        if str(row["Question Type"]).strip()
        else classify_question_type(row["Comment"]),
        axis=1
    )

    df["Comment Depth"] = df.apply(
        lambda row: row["Comment Depth"]
        if str(row["Comment Depth"]).strip()
        else classify_comment_depth(row["Comment"]),
        axis=1
    )

    return df[
        [
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
        ]
    ]


def generate_comment_summary(comment_df):
    if len(comment_df) == 0:
        return pd.DataFrame({
            "Dimension": [
                "Sentiment",
                "Comment depth",
                "Main themes",
                "Questions",
                "Community behaviour"
            ],
            "Result": [
                "No comments",
                "No comments",
                "No themes",
                "No questions",
                "No discussion"
            ]
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

    substantive_share = (
        comment_df["Comment Depth"] == "Substantive / detailed"
    ).mean()

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
# 7. AI strategy functions
# =========================================================

def build_ai_context_from_dashboard(benchmark_df, comments_df=None):
    platform_summary = benchmark_df.groupby(["Competitor", "Platform"]).agg(
        Posts=("Title / Description", "count"),
        Avg_Views=("Views", "mean"),
        Median_Views=("Views", "median"),
        Total_Views=("Views", "sum"),
        Avg_ER=("ER (%)", "mean"),
        Avg_Comments=("Comments", "mean"),
        Total_Comments=("Comments", "sum")
    ).reset_index()

    platform_summary["Avg_Views"] = platform_summary["Avg_Views"].round(0)
    platform_summary["Median_Views"] = platform_summary["Median_Views"].round(0)
    platform_summary["Avg_ER"] = platform_summary["Avg_ER"].round(2)
    platform_summary["Avg_Comments"] = platform_summary["Avg_Comments"].round(1)

    topic_summary = benchmark_df.groupby(
        ["Competitor", "Platform", "Topic / Content Pillar"]
    ).agg(
        Posts=("Title / Description", "count"),
        Avg_Views=("Views", "mean"),
        Avg_ER=("ER (%)", "mean"),
        Avg_Comments=("Comments", "mean")
    ).reset_index()

    topic_summary["Avg_Views"] = topic_summary["Avg_Views"].round(0)
    topic_summary["Avg_ER"] = topic_summary["Avg_ER"].round(2)
    topic_summary["Avg_Comments"] = topic_summary["Avg_Comments"].round(1)

    top_posts_cols = [
        "Competitor",
        "Platform",
        "Title / Description",
        "Views",
        "ER (%)",
        "Comments",
        "Duration Seconds",
        "Hashtags",
        "Topic / Content Pillar",
        "Format"
    ]

    available_top_posts_cols = [c for c in top_posts_cols if c in benchmark_df.columns]

    top_posts = benchmark_df.sort_values("Views", ascending=False).head(30)[
        available_top_posts_cols
    ].copy()

    context = "DASHBOARD DATA SUMMARY\n\n"

    context += "1) Competitor + Platform Performance Summary:\n"
    context += platform_summary.to_string(index=False)
    context += "\n\n"

    context += "2) Topic / Content Pillar Summary:\n"
    context += topic_summary.head(120).to_string(index=False)
    context += "\n\n"

    context += "3) Top Posts by Views:\n"
    context += top_posts.to_string(index=False)
    context += "\n\n"

    if comments_df is not None and len(comments_df) > 0:
        comment_summary_df = comments_df.groupby(["Competitor", "Platform"]).agg(
            Comments_Analysed=("Comment", "count"),
            Positive_Share=("Sentiment", lambda x: (x == "Positive").mean()),
            Neutral_Share=("Sentiment", lambda x: (x == "Neutral").mean()),
            Negative_Share=("Sentiment", lambda x: (x == "Negative").mean()),
            Substantive_Share=("Comment Depth", lambda x: (x == "Substantive / detailed").mean()),
            Question_Share=("Question Type", lambda x: (x != "Not a question").mean())
        ).reset_index()

        for col in [
            "Positive_Share",
            "Neutral_Share",
            "Negative_Share",
            "Substantive_Share",
            "Question_Share"
        ]:
            comment_summary_df[col] = (comment_summary_df[col] * 100).round(1)

        context += "4) Comment Quality Summary:\n"
        context += comment_summary_df.to_string(index=False)
        context += "\n\n"

    return context


def generate_rule_based_strategy_summary(benchmark_df, comments_df=None):
    platform_summary = benchmark_df.groupby(["Competitor", "Platform"]).agg(
        Posts=("Title / Description", "count"),
        Avg_Views=("Views", "mean"),
        Avg_ER=("ER (%)", "mean"),
        Avg_Comments=("Comments", "mean")
    ).reset_index()

    platform_summary["Avg_Views"] = platform_summary["Avg_Views"].round(0)
    platform_summary["Avg_ER"] = platform_summary["Avg_ER"].round(2)
    platform_summary["Avg_Comments"] = platform_summary["Avg_Comments"].round(1)

    top_reach = platform_summary.sort_values("Avg_Views", ascending=False).head(3)
    top_er = platform_summary.sort_values("Avg_ER", ascending=False).head(3)
    top_comments = platform_summary.sort_values("Avg_Comments", ascending=False).head(3)

    topic_summary = benchmark_df.groupby(["Competitor", "Topic / Content Pillar"]).agg(
        Posts=("Title / Description", "count"),
        Avg_Views=("Views", "mean"),
        Avg_ER=("ER (%)", "mean")
    ).reset_index()

    topic_summary["Avg_Views"] = topic_summary["Avg_Views"].round(0)
    topic_summary["Avg_ER"] = topic_summary["Avg_ER"].round(2)

    top_topics = topic_summary.sort_values("Avg_Views", ascending=False).head(5)

    summary = "### Competitor Summary\n\n"

    summary += "**Reach leaders:**\n"
    for _, row in top_reach.iterrows():
        summary += f"- {row['Competitor']} on {row['Platform']} has high average reach, around {row['Avg_Views']:,.0f} views per post.\n"

    summary += "\n**Engagement-rate leaders:**\n"
    for _, row in top_er.iterrows():
        summary += f"- {row['Competitor']} on {row['Platform']} shows stronger engagement, around {row['Avg_ER']}% ER.\n"

    summary += "\n**Discussion proxy:**\n"
    for _, row in top_comments.iterrows():
        summary += f"- {row['Competitor']} on {row['Platform']} generates stronger comment volume, around {row['Avg_Comments']} comments per post.\n"

    summary += "\n**Strong topic opportunities:**\n"
    for _, row in top_topics.iterrows():
        summary += f"- {row['Topic / Content Pillar']} performs strongly for {row['Competitor']}, with average views around {row['Avg_Views']:,.0f}.\n"

    summary += "\n### Marketing Recommendations for Finder\n\n"
    summary += (
        "1. **Prioritise high-intent topics.** Focus on banking, savings, investing, pensions and life-stage finance where users are closer to comparison behaviour.\n\n"
        "2. **Use more human-led hooks.** If individual finance creators outperform Finder, Finder should make educational content feel more personal, question-led and story-driven.\n\n"
        "3. **Connect social content to comparison pages.** Awareness content should link clearly to Finder product pages, especially banking, savings, credit cards and pensions.\n\n"
        "4. **Turn comments into content prompts.** Questions and substantive comments can become future video topics, newsletter sections, or FAQ-led pages.\n\n"
        "5. **Test platform-specific formats.** YouTube should emphasise explainers, comparisons and reviews; TikTok and Instagram should use short pain-point hooks.\n"
    )

    return summary


def generate_ai_strategy_summary_with_openai(
    benchmark_df,
    comments_df,
    openai_api_key,
    model_name
):
    context = build_ai_context_from_dashboard(benchmark_df, comments_df)

    prompt = f"""
You are a senior marketing analytics consultant working on Finder UK's competitor content intelligence dashboard.

Use the dashboard data below to produce a concise but strategic report.

Required output structure:

1. Executive Summary
- 3 bullet points on what the data shows overall.

2. Competitor Summary
- Summarise each major competitor's strengths and weaknesses.
- Mention platform role, topic strengths, reach, engagement, hashtags, video duration, and comment quality where available.

3. Finder Gap Analysis
- Explain where Finder appears weaker than competitors.
- Explain where Finder has an advantage.

4. AI Marketing Recommendations for Finder
- Give 5 actionable recommendations.
- Each recommendation should include:
  - what Finder should do
  - why the dashboard data supports it
  - which platform or topic it applies to

5. Suggested Content Tests
- Give 5 concrete content ideas Finder could test next.
- Include recommended platform, topic, format and example title.

Important rules:
- Do not invent data not shown in the dashboard.
- If a conclusion is based on limited data, say so.
- Use clear business language.
- Keep the output suitable for a management presentation.

Dashboard data:
{context}
"""

    client = OpenAI(api_key=openai_api_key)

    response = client.responses.create(
        model=model_name,
        input=prompt
    )

    return response.output_text


# =========================================================
# 8. ML predictor functions
# =========================================================

def prepare_ml_training_data(benchmark_df, training_mode):
    df = benchmark_df.copy()

    if training_mode == "Finder-only data":
        df = df[df["Competitor"].astype(str).str.lower() == "finder"].copy()

    needed_cols = [
        "Competitor",
        "Platform",
        "Title / Description",
        "Format",
        "Topic / Content Pillar",
        "Views",
        "Comments",
        "ER (%)"
    ]

    df = df[[c for c in needed_cols if c in df.columns]].copy()

    df = df.dropna(
        subset=[
            "Platform",
            "Title / Description",
            "Format",
            "Topic / Content Pillar",
            "Views",
            "Comments",
            "ER (%)"
        ]
    )

    df["Title / Description"] = df["Title / Description"].astype(str)
    df["Platform"] = df["Platform"].astype(str)
    df["Format"] = df["Format"].astype(str)
    df["Topic / Content Pillar"] = df["Topic / Content Pillar"].astype(str)

    df["Views"] = pd.to_numeric(df["Views"], errors="coerce")
    df["Comments"] = pd.to_numeric(df["Comments"], errors="coerce")
    df["ER (%)"] = pd.to_numeric(df["ER (%)"], errors="coerce")

    df = df.dropna(subset=["Views", "Comments", "ER (%)"])

    df = df[df["Views"] >= 0]
    df = df[df["Comments"] >= 0]

    df["log_views"] = np.log1p(df["Views"])
    df["log_comments"] = np.log1p(df["Comments"])

    return df


def train_prediction_model(df, target_col):
    feature_cols = [
        "Platform",
        "Format",
        "Topic / Content Pillar",
        "Title / Description"
    ]

    X = df[feature_cols]
    y = df[target_col]

    if len(df) < 10:
        return None, None

    test_size = 0.25 if len(df) >= 20 else 0.3

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(max_features=300, stop_words="english"),
                "Title / Description"
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                ["Platform", "Format", "Topic / Content Pillar"]
            )
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    min_samples_leaf=2
                )
            )
        ]
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds) if len(y_test) > 1 else None,
        "Train rows": len(X_train),
        "Test rows": len(X_test)
    }

    return model, metrics


def classify_predicted_performance(pred_views, pred_er, platform):
    platform = str(platform).lower()

    if platform == "youtube":
        if pred_views >= 100000 and pred_er >= 2.5:
            return "High potential"
        if pred_views >= 20000 or pred_er >= 1.5:
            return "Medium potential"
        return "Low potential"

    if platform == "tiktok":
        if pred_views >= 100000 and pred_er >= 3:
            return "High potential"
        if pred_views >= 20000 or pred_er >= 2:
            return "Medium potential"
        return "Low potential"

    if platform == "instagram":
        if pred_views >= 100000 and pred_er >= 2:
            return "High potential"
        if pred_views >= 20000 or pred_er >= 1:
            return "Medium potential"
        return "Low potential"

    return "Indicative only"


def predict_content_performance(
    views_model,
    er_model,
    comments_model,
    platform,
    content_format,
    topic,
    title
):
    input_df = pd.DataFrame({
        "Platform": [platform],
        "Format": [content_format],
        "Topic / Content Pillar": [topic],
        "Title / Description": [title]
    })

    pred_log_views = views_model.predict(input_df)[0]
    pred_views = int(max(0, np.expm1(pred_log_views)))

    pred_er = float(max(0, er_model.predict(input_df)[0]))

    pred_log_comments = comments_model.predict(input_df)[0]
    pred_comments = int(max(0, np.expm1(pred_log_comments)))

    tier = classify_predicted_performance(pred_views, pred_er, platform)

    return {
        "Predicted Views": pred_views,
        "Predicted ER (%)": round(pred_er, 2),
        "Predicted Comments": pred_comments,
        "Predicted Tier": tier
    }


# =========================================================
# 9. Sidebar
# =========================================================

st.sidebar.header("YouTube API Collection")

youtube_accounts_text = st.sidebar.text_area(
    "YouTube competitors, one per line",
    value=(
        "Finder | https://www.youtube.com/@FinderUK\n"
        "Nischa | https://www.youtube.com/@nischa\n"
        "Damien Talks Money | https://www.youtube.com/@DamienTalksMoney\n"
        "Monzo | https://www.youtube.com/@MonzoBank"
    ),
    help="Format: Competitor Name | YouTube Channel URL or Handle"
)

youtube_secret_key = get_youtube_api_key()

manual_youtube_key = st.sidebar.text_input(
    "YouTube API Key optional override",
    value="",
    type="password",
    help="Leave blank if YOUTUBE_API_KEY is already saved in Streamlit Secrets."
)

if youtube_secret_key:
    st.sidebar.success("YouTube API key loaded from Streamlit Secrets.")
else:
    st.sidebar.info("No YouTube API key detected. Enter one manually if needed.")

days = st.sidebar.slider(
    "Time Window: recent days to collect",
    min_value=30,
    max_value=365,
    value=90,
    step=30,
    help="The app collects all available YouTube videos published within this time window."
)

top_n_videos = st.sidebar.slider(
    "Top videos to display",
    min_value=5,
    max_value=100,
    value=20,
    step=5,
    help="The app collects all recent videos in the selected time window, then displays the top N videos by views."
)

youtube_max_pages = st.sidebar.slider(
    "YouTube API safety page limit",
    min_value=1,
    max_value=20,
    value=10,
    step=1,
    help="Each page contains up to 50 videos. This prevents excessive API usage."
)

include_shorts = st.sidebar.checkbox(
    "Include Shorts / videos under 60 seconds",
    value=False,
    help="If unchecked, the dashboard excludes videos that are 60 seconds or shorter."
)

st.sidebar.markdown("---")
st.sidebar.header("AI Strategy Summary")

openai_secret_key = get_openai_api_key()

manual_openai_key = st.sidebar.text_input(
    "OpenAI API Key optional override",
    value="",
    type="password",
    help="Leave blank if OPENAI_API_KEY is already saved in Streamlit Secrets."
)

if openai_secret_key:
    st.sidebar.success("OpenAI API key loaded from Streamlit Secrets.")
else:
    st.sidebar.info("No OpenAI API key detected. Rule-based summary still works.")

ai_model_name = st.sidebar.text_input(
    "OpenAI model name",
    value="gpt-5",
    help="Use a model available in your OpenAI account."
)

st.sidebar.markdown("---")
st.sidebar.header("Upload Port 1: Raw Platform Exports")

raw_platform_uploads = st.sidebar.file_uploader(
    "Upload raw TikTok / Instagram / YouTube / other platform exports",
    type=["csv", "xlsx"],
    accept_multiple_files=True,
    help="Required fields: title/caption and views. Likes/comments are recommended."
)

st.sidebar.markdown("---")
st.sidebar.header("Upload Port 2: Prepared Competitor Workbook")

prepared_workbook_upload = st.sidebar.file_uploader(
    "Upload prepared competitor benchmark workbook",
    type=["xlsx"]
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
    help="Required column: Comment."
)

run_button = st.sidebar.button("Run Analysis")


# =========================================================
# 10. Tabs
# =========================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "1. Benchmark Table",
        "2. Performance Summary",
        "3. Topic Analysis",
        "4. Comment Analysis",
        "5. Competitor Comparison",
        "6. AI Strategy Summary",
        "7. ML Predictor",
        "8. Sharing / Deployment Notes"
    ]
)


# =========================================================
# 11. Run analysis
# =========================================================

if run_button:
    st.session_state.pop("benchmark", None)
    st.session_state.pop("comments_df", None)
    st.session_state.pop("comment_summary", None)
    st.session_state.pop("ai_strategy_summary", None)

    all_data = []

    youtube_api_key = manual_youtube_key.strip() if manual_youtube_key else youtube_secret_key

    youtube_accounts = parse_youtube_accounts(youtube_accounts_text)

    if youtube_accounts and youtube_api_key:
        for account in youtube_accounts:
            comp_name = account["competitor"]
            yt_input = account["youtube_input"]

            if not yt_input:
                st.warning(f"{comp_name}: No YouTube channel input provided.")
                continue

            try:
                with st.spinner(f"Resolving YouTube channel for {comp_name}..."):
                    channel_id, error = resolve_youtube_channel_id(yt_input, youtube_api_key)

                if error:
                    st.warning(f"{comp_name}: {error}")
                    continue

                subscriber_count = fetch_youtube_channel_subscriber_count(
                    channel_id=channel_id,
                    api_key=youtube_api_key
                )

                with st.spinner(f"Fetching all recent YouTube videos for {comp_name}..."):
                    yt_df = fetch_youtube_recent_videos(
                        channel_id=channel_id,
                        api_key=youtube_api_key,
                        days=days,
                        include_shorts=include_shorts,
                        max_pages=youtube_max_pages,
                        subscriber_count=subscriber_count
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

    elif youtube_accounts and not youtube_api_key:
        st.warning("No YouTube API key available. Add it in Streamlit Secrets or enter it manually.")

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

    if prepared_workbook_upload is not None:
        try:
            prepared_df = read_prepared_competitor_workbook(prepared_workbook_upload)

            if prepared_df.empty:
                st.warning("Prepared competitor workbook was uploaded, but no valid benchmark tables were found.")
            else:
                all_data.append(prepared_df)

        except Exception as e:
            st.error(f"Prepared competitor workbook upload failed: {e}")

    if all_data:
        benchmark = pd.concat(all_data, ignore_index=True, sort=False)
        benchmark = benchmark.dropna(subset=["Title / Description"], how="any")
        st.session_state["benchmark"] = benchmark

        if fetch_youtube_comments and youtube_api_key:
            with st.spinner("Fetching YouTube comments..."):
                yt_comments_df = fetch_youtube_comments_for_benchmark(
                    benchmark_df=benchmark,
                    api_key=youtube_api_key,
                    max_comments_per_video=max_comments_per_video
                )

            if not yt_comments_df.empty:
                st.session_state["comments_df"] = yt_comments_df
                st.session_state["comment_summary"] = generate_comment_summary(yt_comments_df)
            else:
                st.info("No YouTube comments collected. Comments may be disabled or unavailable.")

    else:
        st.warning(
            "No post/video data collected. Add YouTube API key, upload raw platform exports, "
            "or upload a prepared workbook."
        )

    if comment_uploads:
        uploaded_comment_frames = []

        for uploaded_file in comment_uploads:
            try:
                raw_comments = read_uploaded_file(uploaded_file)
                uploaded_comments_df = standardise_comment_upload(raw_comments)

                inferred_competitor = infer_competitor_from_filename(uploaded_file.name)
                inferred_platform = infer_platform_from_filename(uploaded_file.name)

                uploaded_comments_df["Competitor"] = (
                    uploaded_comments_df["Competitor"]
                    .fillna("")
                    .replace("", inferred_competitor)
                )

                uploaded_comments_df["Platform"] = (
                    uploaded_comments_df["Platform"]
                    .fillna("")
                    .replace("", inferred_platform)
                )

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
# 12. Tab 1: Benchmark Table
# =========================================================

with tab1:
    st.subheader("Standardised Benchmark Tables by Competitor and Platform")

    if benchmark is not None:
        for (competitor, platform), group_df in benchmark.groupby(["Competitor", "Platform"]):
            st.markdown(f"### {competitor} — {platform}")

            display_df = group_df.reset_index(drop=True)
            st.dataframe(display_df, use_container_width=True)

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
        st.dataframe(benchmark.reset_index(drop=True), use_container_width=True)

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
# 13. Tab 2: Performance Summary
# =========================================================

with tab2:
    st.subheader("Performance Summary")

    if benchmark is not None:
        col1, col2, col3, col4 = st.columns(4)

        total_posts = len(benchmark)
        avg_views = benchmark["Views"].mean()
        avg_er = benchmark["ER (%)"].mean()
        avg_comments = benchmark["Comments"].mean()

        col1.metric("Total Posts Collected", f"{total_posts:,.0f}")
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
        st.dataframe(platform_summary, use_container_width=True)

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig_views = px.bar(
                platform_summary,
                x="Competitor",
                y="Avg_Views",
                color="Platform",
                barmode="group",
                title="Average Views by Competitor and Platform"
            )

            st.plotly_chart(fig_views, use_container_width=True)

        with chart_col2:
            fig_er = px.bar(
                platform_summary,
                x="Competitor",
                y="Avg_ER",
                color="Platform",
                barmode="group",
                title="Average Engagement Rate by Competitor and Platform"
            )

            st.plotly_chart(fig_er, use_container_width=True)

        st.markdown("---")
        st.markdown(f"### Top {top_n_videos} Posts by Views")

        top_posts_filtered, top_posts_sort_mode = render_competitor_platform_filters(
            benchmark,
            key_prefix="top_posts"
        )

        if top_posts_filtered.empty:
            st.warning("No posts match the selected filters.")
        else:
            global_top_posts = (
                top_posts_filtered
                .sort_values("Views", ascending=False)
                .head(top_n_videos)
                .reset_index(drop=True)
            )

            st.markdown("#### Full Top Posts Ranking")
            st.dataframe(global_top_posts, use_container_width=True)

            st.markdown("#### Grouped View")

            if top_posts_sort_mode == "By platform":
                for platform, platform_df in global_top_posts.groupby("Platform"):
                    st.markdown(f"##### {platform}")

                    display_platform_posts = (
                        platform_df
                        .sort_values("Views", ascending=False)
                        .reset_index(drop=True)
                    )

                    st.dataframe(display_platform_posts, use_container_width=True)

            else:
                for competitor, competitor_df in global_top_posts.groupby("Competitor"):
                    st.markdown(f"##### {competitor}")

                    display_competitor_posts = (
                        competitor_df
                        .sort_values("Views", ascending=False)
                        .reset_index(drop=True)
                    )

                    st.dataframe(display_competitor_posts, use_container_width=True)

    else:
        st.info("Run analysis to see performance summary.")


# =========================================================
# 14. Tab 3: Topic + Format Analysis
# =========================================================

with tab3:
    st.subheader("Topic / Content Pillar Analysis")

    if benchmark is not None:
        topic_filtered, topic_sort_mode = render_competitor_platform_filters(
            benchmark,
            key_prefix="topic_analysis"
        )

        if topic_filtered.empty:
            st.warning("No data matches the selected filters.")
        else:
            topic_summary = topic_filtered.groupby(
                ["Competitor", "Platform", "Topic / Content Pillar"]
            ).agg(
                Posts=("Title / Description", "count"),
                Avg_Views=("Views", "mean"),
                Avg_ER=("ER (%)", "mean"),
                Avg_Comments=("Comments", "mean")
            ).reset_index()

            topic_summary["Avg_Views"] = topic_summary["Avg_Views"].round(0)
            topic_summary["Avg_ER"] = topic_summary["Avg_ER"].round(2)
            topic_summary["Avg_Comments"] = topic_summary["Avg_Comments"].round(1)

            st.markdown("### Topic / Content Pillar Summary")

            if topic_sort_mode == "By platform":
                for platform, platform_df in topic_summary.groupby("Platform"):
                    st.markdown(f"#### {platform}")

                    display_topic = (
                        platform_df
                        .sort_values(["Competitor", "Avg_Views"], ascending=[True, False])
                        .reset_index(drop=True)
                    )

                    st.dataframe(display_topic, use_container_width=True)

            else:
                for competitor, competitor_df in topic_summary.groupby("Competitor"):
                    st.markdown(f"#### {competitor}")

                    display_topic = (
                        competitor_df
                        .sort_values(["Platform", "Avg_Views"], ascending=[True, False])
                        .reset_index(drop=True)
                    )

                    st.dataframe(display_topic, use_container_width=True)

            st.markdown("---")
            st.markdown("### Format Summary")

            format_summary = topic_filtered.groupby(
                ["Competitor", "Platform", "Format"]
            ).agg(
                Posts=("Title / Description", "count"),
                Avg_Views=("Views", "mean"),
                Avg_ER=("ER (%)", "mean"),
                Avg_Comments=("Comments", "mean")
            ).reset_index()

            format_summary["Avg_Views"] = format_summary["Avg_Views"].round(0)
            format_summary["Avg_ER"] = format_summary["Avg_ER"].round(2)
            format_summary["Avg_Comments"] = format_summary["Avg_Comments"].round(1)

            if topic_sort_mode == "By platform":
                for platform, platform_df in format_summary.groupby("Platform"):
                    st.markdown(f"#### {platform}")

                    display_format = (
                        platform_df
                        .sort_values(["Competitor", "Avg_Views"], ascending=[True, False])
                        .reset_index(drop=True)
                    )

                    st.dataframe(display_format, use_container_width=True)

            else:
                for competitor, competitor_df in format_summary.groupby("Competitor"):
                    st.markdown(f"#### {competitor}")

                    display_format = (
                        competitor_df
                        .sort_values(["Platform", "Avg_Views"], ascending=[True, False])
                        .reset_index(drop=True)
                    )

                    st.dataframe(display_format, use_container_width=True)

    else:
        st.info("Run analysis to see topic analysis.")


# =========================================================
# 15. Tab 4: Comment Analysis
# =========================================================

with tab4:
    st.subheader("Comment Analysis")

    if comments_df is not None:
        st.markdown("### Comment-Level Coding Tables by Competitor and Platform")

        for (competitor, platform), group_df in comments_df.groupby(["Competitor", "Platform"]):
            st.markdown(f"#### {competitor} — {platform}")

            display_comments = group_df.reset_index(drop=True)
            st.dataframe(display_comments, use_container_width=True)

            csv_comments_group = display_comments.to_csv(index=False).encode("utf-8")

            st.download_button(
                f"Download {competitor}_{platform}_comments.csv",
                data=csv_comments_group,
                file_name=f"{competitor}_{platform}_comments.csv".replace(" ", "_"),
                mime="text/csv",
                key=f"download_comments_{competitor}_{platform}"
            )

        st.markdown("---")
        st.markdown("### Full Combined Comment-Level Coding Table")
        st.dataframe(comments_df.reset_index(drop=True), use_container_width=True)

        full_comments_csv = comments_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download full combined comment-level table as CSV",
            data=full_comments_csv,
            file_name="full_combined_comment_level_analysis.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.markdown("### Overall Comment Summary")
        st.dataframe(comment_summary, use_container_width=True)

        st.markdown("---")
        st.markdown("### Comment Summary by Competitor and Platform")

        comment_group_summary = comments_df.groupby(["Competitor", "Platform"]).agg(
            Comments_Analysed=("Comment", "count"),
            Avg_Comment_Likes=("Comment Likes", "mean"),
            Positive_Share=("Sentiment", lambda x: (x == "Positive").mean()),
            Neutral_Share=("Sentiment", lambda x: (x == "Neutral").mean()),
            Negative_Share=("Sentiment", lambda x: (x == "Negative").mean()),
            Substantive_Share=("Comment Depth", lambda x: (x == "Substantive / detailed").mean()),
            Question_Share=("Question Type", lambda x: (x != "Not a question").mean())
        ).reset_index()

        comment_group_summary["Avg_Comment_Likes"] = comment_group_summary["Avg_Comment_Likes"].round(1)
        comment_group_summary["Positive_Share"] = (comment_group_summary["Positive_Share"] * 100).round(1)
        comment_group_summary["Neutral_Share"] = (comment_group_summary["Neutral_Share"] * 100).round(1)
        comment_group_summary["Negative_Share"] = (comment_group_summary["Negative_Share"] * 100).round(1)
        comment_group_summary["Substantive_Share"] = (comment_group_summary["Substantive_Share"] * 100).round(1)
        comment_group_summary["Question_Share"] = (comment_group_summary["Question_Share"] * 100).round(1)

        st.dataframe(comment_group_summary, use_container_width=True)

        st.markdown("---")
        st.markdown("### Comment Distribution")

        col1, col2 = st.columns(2)

        with col1:
            sentiment_chart = comments_df.groupby(
                ["Competitor", "Platform", "Sentiment"]
            ).size().reset_index(name="Count")

            fig_sentiment = px.bar(
                sentiment_chart,
                x="Sentiment",
                y="Count",
                color="Competitor",
                barmode="group",
                title="Comment Sentiment Distribution by Competitor"
            )

            st.plotly_chart(fig_sentiment, use_container_width=True)

        with col2:
            depth_chart = comments_df.groupby(
                ["Competitor", "Platform", "Comment Depth"]
            ).size().reset_index(name="Count")

            fig_depth = px.bar(
                depth_chart,
                x="Comment Depth",
                y="Count",
                color="Competitor",
                barmode="group",
                title="Comment Depth Distribution by Competitor"
            )

            st.plotly_chart(fig_depth, use_container_width=True)

        theme_chart = comments_df.groupby(
            ["Competitor", "Platform", "Theme"]
        ).size().reset_index(name="Count")

        fig_theme = px.bar(
            theme_chart,
            x="Theme",
            y="Count",
            color="Competitor",
            barmode="group",
            title="Main Comment Themes by Competitor"
        )

        st.plotly_chart(fig_theme, use_container_width=True)

    else:
        st.info(
            "Run YouTube analysis with automatic comment fetching, "
            "or upload TikTok / Instagram / other comment files."
        )


# =========================================================
# 16. Tab 5: Competitor Comparison
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
                Substantive_Share=("Comment Depth", lambda x: (x == "Substantive / detailed").mean()),
                Question_Share=("Question Type", lambda x: (x != "Not a question").mean())
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
                if value < finder_value * 0.8:
                    return "Weaker than Finder"

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

        st.dataframe(comparison, use_container_width=True)

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
# 17. Tab 6: AI Strategy Summary
# =========================================================

with tab6:
    st.subheader("AI Strategy Summary")

    st.markdown(
        """
        This section turns dashboard outputs into a strategy layer:

        **Dashboard data → AI competitor summary → AI marketing recommendations**
        """
    )

    if benchmark is None:
        st.info(
            "Run analysis first. The AI summary needs benchmark data from YouTube API, uploaded platform exports, or a prepared workbook."
        )
    else:
        ai_context_preview = build_ai_context_from_dashboard(
            benchmark_df=benchmark,
            comments_df=comments_df
        )

        with st.expander("Preview dashboard data sent to AI"):
            st.text(ai_context_preview[:8000])

        summary_mode = st.radio(
            "Summary mode",
            options=[
                "AI-generated summary",
                "Rule-based automatic summary"
            ],
            horizontal=True
        )

        final_openai_key = manual_openai_key.strip() if manual_openai_key else openai_secret_key

        if summary_mode == "AI-generated summary":
            if not final_openai_key:
                st.warning(
                    "No OpenAI API key found. Add OPENAI_API_KEY in Streamlit Secrets, "
                    "or choose rule-based automatic summary."
                )
            else:
                if st.button("Generate AI competitor summary and marketing recommendations"):
                    with st.spinner("Generating AI strategy summary..."):
                        try:
                            ai_summary = generate_ai_strategy_summary_with_openai(
                                benchmark_df=benchmark,
                                comments_df=comments_df,
                                openai_api_key=final_openai_key,
                                model_name=ai_model_name
                            )

                            st.session_state["ai_strategy_summary"] = ai_summary

                        except Exception as e:
                            st.error(f"AI summary generation failed: {e}")

        else:
            if st.button("Generate rule-based automatic summary"):
                summary = generate_rule_based_strategy_summary(
                    benchmark_df=benchmark,
                    comments_df=comments_df
                )

                st.session_state["ai_strategy_summary"] = summary

        if "ai_strategy_summary" in st.session_state:
            st.markdown("---")
            st.markdown("### Generated Strategy Summary")
            st.markdown(st.session_state["ai_strategy_summary"])

            st.download_button(
                "Download strategy summary as TXT",
                data=st.session_state["ai_strategy_summary"].encode("utf-8"),
                file_name="ai_strategy_summary.txt",
                mime="text/plain"
            )


# =========================================================
# 18. Tab 7: ML Predictor
# =========================================================

with tab7:
    st.subheader("ML Content Performance Predictor")

    st.markdown(
        """
        This predictor trains a lightweight machine-learning model on the benchmark table.  
        It predicts expected **views**, **engagement rate**, and **comments** for a new content idea.

        Model used: **Random Forest Regressor**  
        Features used: **Platform, Format, Topic / Content Pillar, Title / Description**
        """
    )

    if benchmark is not None:
        training_mode = st.radio(
            "Training data mode",
            [
                "All benchmark data",
                "Finder-only data"
            ],
            help=(
                "All benchmark data uses Finder + competitors. "
                "Finder-only data uses only Finder's own historical content."
            )
        )

        ml_df = prepare_ml_training_data(benchmark, training_mode)

        if training_mode == "Finder-only data":
            st.info(
                "Finder-only mode is more brand-specific, but it requires enough Finder historical rows. "
                "If Finder data is limited, use All benchmark data for a broader competitor-informed model."
            )
        else:
            st.info(
                "All benchmark mode uses Finder and competitor content together. "
                "This is better for competitor-informed planning when Finder-only data is limited."
            )

        st.markdown("### Training data preview")
        st.write(f"Rows available for training: **{len(ml_df)}**")
        st.dataframe(ml_df.head(20), use_container_width=True)

        if len(ml_df) < 10:
            st.warning(
                "Not enough rows to train a reliable model. "
                "Please upload or collect at least 10 content rows for the selected training mode."
            )
        else:
            views_model, views_metrics = train_prediction_model(ml_df, "log_views")
            er_model, er_metrics = train_prediction_model(ml_df, "ER (%)")
            comments_model, comments_metrics = train_prediction_model(ml_df, "log_comments")

            if views_model is None or er_model is None or comments_model is None:
                st.warning("Model training failed. Please upload more benchmark data.")
            else:
                st.markdown("### Model diagnostics")

                metrics_df = pd.DataFrame(
                    [
                        {
                            "Target": "Views",
                            "MAE": round(np.expm1(views_metrics["MAE"]), 0),
                            "R2": round(views_metrics["R2"], 3)
                            if views_metrics["R2"] is not None
                            else None,
                            "Train rows": views_metrics["Train rows"],
                            "Test rows": views_metrics["Test rows"]
                        },
                        {
                            "Target": "ER (%)",
                            "MAE": round(er_metrics["MAE"], 2),
                            "R2": round(er_metrics["R2"], 3)
                            if er_metrics["R2"] is not None
                            else None,
                            "Train rows": er_metrics["Train rows"],
                            "Test rows": er_metrics["Test rows"]
                        },
                        {
                            "Target": "Comments",
                            "MAE": round(np.expm1(comments_metrics["MAE"]), 0),
                            "R2": round(comments_metrics["R2"], 3)
                            if comments_metrics["R2"] is not None
                            else None,
                            "Train rows": comments_metrics["Train rows"],
                            "Test rows": comments_metrics["Test rows"]
                        }
                    ]
                )

                st.dataframe(metrics_df, use_container_width=True)

                st.markdown("### Predict a new content idea")

                col1, col2 = st.columns(2)

                with col1:
                    pred_platform = st.selectbox(
                        "Platform",
                        sorted(benchmark["Platform"].dropna().astype(str).unique().tolist())
                    )

                    pred_format = st.selectbox(
                        "Format",
                        sorted(benchmark["Format"].dropna().astype(str).unique().tolist())
                    )

                with col2:
                    pred_topic = st.selectbox(
                        "Topic / Content Pillar",
                        sorted(benchmark["Topic / Content Pillar"].dropna().astype(str).unique().tolist())
                    )

                pred_title = st.text_area(
                    "Proposed title / description",
                    value="Are you missing out on free money in your pension?"
                )

                if st.button("Predict content performance"):
                    result = predict_content_performance(
                        views_model=views_model,
                        er_model=er_model,
                        comments_model=comments_model,
                        platform=pred_platform,
                        content_format=pred_format,
                        topic=pred_topic,
                        title=pred_title
                    )

                    st.markdown("### Prediction result")

                    result_col1, result_col2, result_col3, result_col4 = st.columns(4)

                    result_col1.metric("Predicted Views", f"{result['Predicted Views']:,.0f}")
                    result_col2.metric("Predicted ER", f"{result['Predicted ER (%)']:.2f}%")
                    result_col3.metric("Predicted Comments", f"{result['Predicted Comments']:,.0f}")
                    result_col4.metric("Predicted Tier", result["Predicted Tier"])

                    st.info(
                        "This prediction is indicative only. Accuracy depends on the size and quality of the benchmark data."
                    )

                st.markdown("### How to interpret the two modes")

                st.markdown(
                    """
                    **All benchmark data** is best for competitor-informed planning.  
                    It learns from Finder and competitor content together, so it can identify broader patterns across topics, formats and platforms.

                    **Finder-only data** is best for brand-specific optimisation.  
                    It only learns from Finder's own historical content, so it is more specific to Finder, but it needs enough Finder rows to be stable.
                    """
                )

    else:
        st.info(
            "Run analysis first or upload a prepared competitor workbook. "
            "The predictor needs benchmark data before it can train a model."
        )


# =========================================================
# 19. Tab 8: Notes
# =========================================================

with tab8:
    st.subheader("Sharing / Deployment Notes")

    st.markdown(
        """
        ### YouTube Collection Logic

        The app collects all available YouTube videos published within the selected time window,
        subject to the YouTube API safety page limit.

        Example:

        - Time Window = 90 days
        - YouTube API safety page limit = 10
        - Each page contains up to 50 videos

        This means the app can collect up to 500 recent videos per competitor within the selected period,
        then rank them by views and display the selected Top N videos.

        ### Benchmark Fields

        The benchmark table includes:

        - Views
        - Likes
        - Comments
        - Duration Seconds
        - Hashtags
        - Follower / Subscriber Count
        - Date Collected
        - ER (%)

        For YouTube, public API data provides views, likes, comments, duration, hashtags from title/description, and subscriber count.

        Hashtags are author-provided hashtags extracted automatically from video title and description.

        Engagement Rate is calculated as:

        ```
        ER (%) = (Likes + Comments) / Views * 100
        ```

        ### Upload Port Design

        **Upload Port 1: Raw Platform Exports**  
        Use this for single-platform exports, such as TikTok, Instagram, or YouTube post-level data.  
        Required fields: title/caption and views. Likes/comments are recommended.

        **Upload Port 2: Prepared Competitor Benchmark Workbook**  
        Use this for a prepared workbook where each competitor has a sheet.  
        The table header should contain: Platform, Post URL / Thread, Title / Description, Views, Likes, Comments.

        **Upload Port 3: Comment Files**  
        Use this for TikTok / Instagram / other platform comment files.  
        Required field: Comment.

        ### API Keys in Streamlit Secrets

        ```
        YOUTUBE_API_KEY = "your_youtube_api_key_here"
        OPENAI_API_KEY = "your_openai_api_key_here"
        ```

        Do not put API keys directly into `app.py` or GitHub.

        ### AI Strategy Summary

        The AI strategy tab converts dashboard data into:
        - executive summary
        - competitor summary
        - Finder gap analysis
        - marketing recommendations
        - suggested content tests

        ### ML Predictor

        The predictor uses a **Random Forest Regressor**.

        Two training modes are available:

        1. **All benchmark data**  
           Uses Finder + competitor data. Best for competitor-informed planning.

        2. **Finder-only data**  
           Uses only Finder's own historical data. Best for brand-specific optimisation when enough Finder data exists.

        ### Recommended YouTube input format

        ```
        Finder | https://www.youtube.com/@FinderUK
        Monzo | https://www.youtube.com/@MonzoBank
        Nischa | https://www.youtube.com/@nischa
        Damien Talks Money | https://www.youtube.com/@DamienTalksMoney
        ```
        """
    )