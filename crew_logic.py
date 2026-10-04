import os
import re
import json
import time
import tempfile
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel
from crewai import Agent, Crew, Process, Task, LLM

from tools import seo_search, shopify_product, review_analyzer

load_dotenv()

# Busy ya band model ho to code khud agla model try karta hai
MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3-flash-preview"]
RETRY_WORDS = (
    "503", "unavailable", "overloaded", "high demand",
    "404", "not_found", "no longer available",
    "429", "resource_exhausted", "rate limit",
)


# ---------- Structured outputs (app.py ko yehi format chahiye) ----------
class ReviewOut(BaseModel):
    positive: int
    neutral: int
    negative: int
    pros: List[str]
    cons: List[str]


class SeoOut(BaseModel):
    seo_title: str
    seo_description: str
    keywords: List[str]


class SocialOut(BaseModel):
    instagram: str
    facebook: str
    x_post: str


def _get(task, model):
    """Task ka output pydantic object mein nikalta hai (JSON fallback ke saath)."""
    out = task.output
    if out is None:
        return None
    if getattr(out, "pydantic", None):
        return out.pydantic
    raw = getattr(out, "raw", "") or ""
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            return model(**json.loads(m.group(0)))
        except Exception:
            return None
    return None


def _run_once(model_name, name, desc, tone, reviews_text, reviews_path, store_line, seo_tools):
    llm = LLM(model=f"gemini/{model_name}", api_key=os.getenv("GEMINI_API_KEY"))

    # ---------- Agents ----------
    review_agent = Agent(
        role="Customer Sentiment Analyst",
        goal="Analyze customer reviews and extract pros, cons and sentiment.",
        backstory="You are an expert product analyst who finds real insights in reviews.",
        tools=[review_analyzer],
        llm=llm, verbose=True, allow_delegation=False,
    )
    seo_agent = Agent(
        role="SEO Content Specialist",
        goal="Write an SEO-friendly product description from review insights and keywords.",
        backstory="You are an e-commerce SEO writer whose descriptions rank in search.",
        tools=seo_tools,
        llm=llm, verbose=True, allow_delegation=False,
    )
    social_agent = Agent(
        role="Social Media Marketer",
        goal="Create engaging social media posts from the product description.",
        backstory="You are a creative marketer who writes for Instagram, Facebook and X.",
        llm=llm, verbose=True, allow_delegation=False,
    )

    # ---------- Tasks ----------
    review_task = Task(
        description=(
            f"Product: {name}\n"
            f"Use the Review Analyzer tool with this file path: {reviews_path}\n"
            "If the tool says the file is not in CSV format (columns: review, rating), "
            "analyze the review text below directly instead.\n\n"
            f"REVIEWS:\n{reviews_text}\n\n"
            "Return sentiment as whole-number percentages (positive + neutral + negative = 100), "
            "plus 2 to 4 short pros and 2 to 4 short cons."
        ),
        expected_output="Sentiment percentages, pros list and cons list.",
        agent=review_agent,
        output_pydantic=ReviewOut,
    )
    seo_task = Task(
        description=(
            f"Product: {name}\nShort description: {desc}\n{store_line}\n"
            f"Brand tone: {tone}\n"
            "Use the review pros to highlight strengths and address cons honestly. "
            "If tools are available, research keywords and check the store product data.\n"
            "Write: an SEO title (under 70 characters), a product description "
            "(120 to 150 words), and 5 keywords."
        ),
        expected_output="SEO title, description and keywords.",
        agent=seo_agent,
        context=[review_task],
        output_pydantic=SeoOut,
    )
    social_task = Task(
        description=(
            f"Product: {name}\nBrand tone: {tone}\n"
            "Using the SEO description, write 3 posts: Instagram (with hashtags), "
            "Facebook, and X (under 280 characters)."
        ),
        expected_output="Instagram, Facebook and X posts.",
        agent=social_agent,
        context=[seo_task],
        output_pydantic=SocialOut,
    )

    crew = Crew(
        agents=[review_agent, seo_agent, social_agent],
        tasks=[review_task, seo_task, social_task],
        process=Process.sequential,
        verbose=True,
    )
    crew.kickoff()

    r = _get(review_task, ReviewOut)
    s = _get(seo_task, SeoOut)
    p = _get(social_task, SocialOut)
    if not (r and s and p):
        raise RuntimeError("AI output format sahi nahi aaya. Dobara Generate dabao.")

    return {
        "sentiment": {"positive": r.positive, "neutral": r.neutral, "negative": r.negative},
        "pros": r.pros,
        "cons": r.cons,
        "seo_title": s.seo_title,
        "seo_description": s.seo_description,
        "keywords": s.keywords,
        "posts": {
            "Instagram": p.instagram,
            "Facebook": p.facebook,
            "X (Twitter)": p.x_post,
        },
    }


def run_crew(name, desc, reviews, tone="Professional"):
    """app.py isko aise call karta hai: run_crew(name, desc, reviews, tone)
    Return: dict (sentiment, pros, cons, seo_title, seo_description, keywords, posts)
    """
    # Reviews ko temp file mein save karte hain taake Review Analyzer tool use ho sake
    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
    tmp.write(reviews)
    tmp.close()
    reviews_path = tmp.name
    reviews_text = reviews[:6000]

    # Description mein store URL ho to Shopify tool use hoga
    m = re.search(r"https?://\S+", desc or "")
    store_line = f"Store URL: {m.group(0)}" if m else "Store URL: not provided"

    # Serper key na ho to SEO search skip (app crash nahi hota)
    seo_tools = [shopify_product]
    if os.getenv("SERPER_API_KEY"):
        seo_tools.insert(0, seo_search)

    last_error = None
    try:
        for model_name in MODELS:
            for attempt in range(2):  # har model ko 2 baar try
                try:
                    return _run_once(
                        model_name, name, desc, tone,
                        reviews_text, reviews_path, store_line, seo_tools,
                    )
                except Exception as e:
                    last_error = e
                    msg = str(e).lower()
                    if any(w in msg for w in RETRY_WORDS):
                        time.sleep(3)
                        # 404/band model ho to dobara usi par try karna bekaar hai
                        if "404" in msg or "not_found" in msg or "no longer available" in msg:
                            break
                        continue
                    raise  # koi aur error ho to seedha dikhao
        raise RuntimeError(
            "Google AI abhi bohat busy hai. 1 minute baad dobara Generate dabao. "
            f"(Last error: {last_error})"
        )
    finally:
        try:
            os.remove(reviews_path)
        except OSError:
            pass
