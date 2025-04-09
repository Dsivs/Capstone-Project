import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction import text as sklearn_text
from sklearn.metrics import pairwise

# U.S. State Tax Rates
_STATE_TAX_RATES = {
    "AL": 0.0400, "AK": 0.0000, "AZ": 0.0560, "AR": 0.0650, "CA": 0.0725,
    "CO": 0.0290, "CT": 0.0635, "DE": 0.0000, "FL": 0.0600, "GA": 0.0400,
    "HI": 0.0400, "ID": 0.0600, "IL": 0.0625, "IN": 0.0700, "IA": 0.0600,
    "KS": 0.0650, "KY": 0.0600, "LA": 0.0445, "ME": 0.0550, "MD": 0.0600,
    "MA": 0.0625, "MI": 0.0600, "MN": 0.0688, "MS": 0.0700, "MO": 0.0423,
    "MT": 0.0000, "NE": 0.0550, "NV": 0.0685, "NH": 0.0000, "NJ": 0.0663,
    "NM": 0.0513, "NY": 0.0400, "NC": 0.0475, "ND": 0.0500, "OH": 0.0575,
    "OK": 0.0450, "OR": 0.0000, "PA": 0.0600, "RI": 0.0700, "SC": 0.0600,
    "SD": 0.0450, "TN": 0.0700, "TX": 0.0625, "UT": 0.0610, "VT": 0.0600,
    "VA": 0.0530, "WA": 0.0650, "WV": 0.0600, "WI": 0.0500, "WY": 0.0400,
    "DC": 0.0600, 
}  # fmt: skip

_PAYMENT_DAYS = {
    "NET 30 DAYS": 30,
    "NET 45 DAYS": 45,
    "NET 60 DAYS": 60,
    "NET 15 DAYS": 15,
    "DUE ON RECEIPT": 0,
    "COD": 0,
    "2/10 NET 30": 30,
}

_US_TERRITORIES = {"PR", "MP", "PW", "VI", "AS", "MH", "GU", "FM"}


def _extract_state(address: str) -> str | None:
    """Extract state abbreviation from a U.S. address."""
    match = re.search(r"\b([A-Z]{2})\s+\d{5}\b", address)
    return match.group(1) if match else None


def process_invoice(
    invoices: list[dict],
    phantom_threshold: float = 0.05,
    duplicate_threshold: float = 0.95,
) -> pd.DataFrame:
    """Process invoice with feature engineering for anomaly detection.

    Args:
        invoices (list[dict]): List of invoice JSON objects
        phantom_threshold (float): Similarity threshold to flag
            phantom items
        duplicate_threshold (float): Similarity threshold to flag
            duplicate invoices

    Returns:
        pd.DataFrame: Engineered dataset with anomaly-related features.
            Return an empty DataFrame if the invoice list is empty
    """
    if not invoices:
        return pd.DataFrame()

    # Calculate invoice similarities
    vectorizer = sklearn_text.TfidfVectorizer()
    invoice_texts = [str(invoice) for invoice in invoices]
    tfidf_matrix = vectorizer.fit_transform(invoice_texts)
    similarity_matrix = pairwise.cosine_similarity(tfidf_matrix)
    np.fill_diagonal(similarity_matrix, -1)  # Exclude self-similarity
    max_invoice_similarity = similarity_matrix.max(axis=1)

    records = []
    for i, (invoice, similarity) in enumerate(zip(invoices, max_invoice_similarity)):
        label = invoice.get("label", 0)
        extraction_fields = {
            entry["field"]: entry["value"] for entry in invoice.get("extractions", [])
        }

        # Grab and remove line_details from extraction_fields
        line_items = extraction_fields.pop("line_details", [])
        for item in line_items:
            record = {**extraction_fields, **item}
            record["invoice_idx"] = i
            record["invoice_similarity"] = similarity
            record["is_anomalous"] = label
            records.append(record)

    df = pd.DataFrame(records)

    # Convert numeric fields
    df["grand_total"] = pd.to_numeric(df["grand_total"], errors="coerce")
    df["tax"] = pd.to_numeric(df["tax"], errors="coerce")
    df["line_qty"] = pd.to_numeric(df["line_qty"], errors="coerce")
    df["line_tax"] = pd.to_numeric(df["line_tax"], errors="coerce")
    df["line_total"] = pd.to_numeric(df["line_total"], errors="coerce")

    # Convert date fields
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["due_date"] = pd.to_datetime(df["due_date"], errors="coerce")

    # State and tax info
    df["state"] = df["merchant_address"].apply(lambda x: _extract_state(str(x)))
    df["expected_tax_rate"] = df["state"].apply(
        lambda x: _STATE_TAX_RATES.get(x, 0.0) if x not in _US_TERRITORIES else 0.0
    )
    df["expected_tax"] = df["expected_tax_rate"] * (df["grand_total"] - df["tax"])
    try:
        df["actual_tax_rate"] = df["tax"] / (df["grand_total"] - df["tax"])
    except ZeroDivisionError:
        df["actual_tax_rate"] = np.nan
    df["tax_mismatch_flag"] = (df["expected_tax"] - df["tax"]).abs() > 0.01 * df[
        "grand_total"
    ]

    # Line similarity check
    vectorizer = sklearn_text.TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["line_description"].fillna(""))
    similarity_matrix = pairwise.cosine_similarity(tfidf_matrix)
    df["avg_description_similarity"] = similarity_matrix.mean(axis=1)
    df["phantom_item_flag"] = df["avg_description_similarity"] < phantom_threshold

    # Invoice similarity check
    df["duplicate_invoice_flag"] = df["invoice_similarity"] > duplicate_threshold

    # Payment days check
    df["payment_terms_numeric"] = df["payment_terms"].map(_PAYMENT_DAYS)
    df["invoice_age"] = (df["due_date"] - df["invoice_date"]).dt.days
    df["invoice_age_mismatch"] = df["invoice_age"] != df["payment_terms_numeric"]

    # Negative quantity check
    df["negative_qty_flag"] = df["line_qty"] < 0

    # Merchant mismatch check
    df["merchant_mismatch_flag"] = df["merchant_branch"] != df["merchant_chain"]

    # Duplicate product check
    df["duplicate_product_flag"] = df.groupby("invoice_idx")[
        "line_description"
    ].transform(lambda x: x.duplicated(keep=False))

    # Aggregate on the invoice basis
    invoice_df = (
        df.groupby("invoice_idx")
        .agg(
            {
                "merchant": "first",
                "merchant_branch": "first",
                "merchant_chain": "first",
                "po_number": "first",
                "payment_method": "first",
                "payment_terms_numeric": "first",
                "invoice_age": "first",
                "invoice_age_mismatch": "any",
                "country": "first",
                "currency": "first",
                "grand_total": "first",
                "tax": "first",
                "actual_tax_rate": "first",
                "expected_tax_rate": "first",
                "expected_tax": "first",
                "tax_mismatch_flag": "any",
                "phantom_item_flag": "any",
                "negative_qty_flag": "any",
                "duplicate_product_flag": "any",
                "merchant_mismatch_flag": "any",
                "duplicate_invoice_flag": "any",
                "avg_description_similarity": "mean",
                "invoice_similarity": "first",
                "line_total": "sum",
                "line_qty": "sum",
                "invoice_date": "first",
                "due_date": "first",
                "merchant_address": "first",
                "state": "first",
                "is_anomalous": "max",
            }
        )
        .reset_index()
        .drop(columns=["invoice_idx"])
    )

    return invoice_df
