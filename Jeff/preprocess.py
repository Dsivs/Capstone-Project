import json
import numpy as np
import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- U.S. State Tax Rates ---
STATE_TAX_RATES = {
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
    "DC": 0.0600
}

# Helper functions
def extract_numeric(x):
    """Convert string to numeric, return None if fails."""
    try:
        return float(x)
    except:
        return None

def extract_state(address):
    """Extract state abbreviation from a U.S. address."""
    match = re.search(r'\b([A-Z]{2})\s+\d{5}\b', address)
    return match.group(1) if match else None

def process_invoice(invoices, phantom_threshold=0.2):
    """
    Process invoice data with feature engineering for fraud detection.

    Parameters:
        invoices (list): List of invoice JSON objects.
        phantom_threshold (float): Similarity threshold to flag phantom items.

    Returns:
        pd.DataFrame: Engineered dataset with fraud-related features.
    """
    records = []
    for invoice in invoices:
        label = invoice.get('label', 0)  # Default fraud label to 0 if missing
        invoice_info = {field['field']: field['value'] for field in invoice['extractions']}
        line_items = invoice_info.pop('line_details', [])
        for item in line_items:
            record = {**invoice_info, **item}
            record['is_fraud'] = label
            records.append(record)

    df = pd.DataFrame(records)

    # Convert numeric fields
    df["grand_total"] = df["grand_total"].apply(extract_numeric)
    df["tax"] = df["tax"].apply(extract_numeric)
    df["line_qty"] = df["line_qty"].apply(extract_numeric)
    df["line_tax"] = df["line_tax"].apply(extract_numeric)
    df["line_total"] = df["line_total"].apply(extract_numeric)

    # Convert date fields
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["due_date"] = pd.to_datetime(df["due_date"], errors="coerce")

    # Extract state from address
    df["state"] = df["merchant_address"].apply(lambda x: extract_state(str(x)))

    # Add expected tax rate column
    df["expected_tax_rate"] = df["state"].map(STATE_TAX_RATES)

    # Compute actual tax rate
    df["actual_tax_rate"] = df["tax"] / (df["grand_total"] - df["tax"] + 1e-5)

    # Compute expected tax
    df["expected_tax"] = df["expected_tax_rate"] * (df["grand_total"] - df["tax"])

    # Flag mismatched tax rates
    df["tax_mismatch_flag"] = (df["expected_tax"] - df["tax"]).abs() > 0.01 * df["grand_total"]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["line_description"].fillna(""))
    similarity_matrix = cosine_similarity(tfidf_matrix)
    df["avg_description_similarity"] = similarity_matrix.mean(axis=1)
    df["phantom_item_flag"] = df["avg_description_similarity"] < phantom_threshold

    return df
