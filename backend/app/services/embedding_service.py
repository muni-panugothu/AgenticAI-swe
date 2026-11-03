import os

from dotenv import load_dotenv
from google import genai


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# GOOGLE API CONFIGURATION
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


if not GOOGLE_API_KEY:

    raise ValueError(
        "GOOGLE_API_KEY is not configured in .env"
    )


# ============================================================
# GOOGLE CLIENT
# ============================================================

client = genai.Client(
    api_key=GOOGLE_API_KEY
)


# ============================================================
# EMBEDDING CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "gemini-embedding-001"


# ============================================================
# CREATE EMBEDDING
# ============================================================

def create_embedding(text):

    try:

        if not text or not text.strip():

            raise ValueError(
                "Text cannot be empty"
            )


        response = client.models.embed_content(

            model=EMBEDDING_MODEL,

            contents=text

        )


        embedding = response.embeddings[0].values


        print(
            f"[Embedding generated] "
            f"Dimensions: {len(embedding)}"
        )


        return embedding


    except Exception as e:

        print(
            f"[Embedding error]: {e}"
        )

        raise