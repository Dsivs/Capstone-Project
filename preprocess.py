import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import text as skl_text
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


def _calculate_invoice_similarity(
    invoices: list[dict],
    vectorizer: skl_text.TfidfVectorizer | None = None,
    train_vecs: sparse.spmatrix | None = None,
    is_train: bool = True,
) -> tuple[np.ndarray, skl_text.TfidfVectorizer | None, sparse.spmatrix | None]:
    invoice_texts = [str(invoice) for invoice in invoices]
    vectorizer_ = None
    train_vecs_ = None

    if is_train:
        vectorizer_ = skl_text.TfidfVectorizer()
        train_vecs_ = vectorizer_.fit_transform(invoice_texts)

        similarity_matrix = pairwise.cosine_similarity(train_vecs_)
        np.fill_diagonal(similarity_matrix, -1)
        max_similarity = similarity_matrix.max(axis=1)

        return max_similarity, vectorizer_, train_vecs_
    else:
        if vectorizer is None or train_vecs is None:
            raise ValueError(
                "Must provide vectorizer and train vectors when `is_train=False`."
            )

        test_vecs = vectorizer.transform(invoice_texts)

        similarity_test_train = pairwise.cosine_similarity(test_vecs, train_vecs)
        max_similarity_train = similarity_test_train.max(axis=1)

        similarity_test_test = pairwise.cosine_similarity(test_vecs)
        np.fill_diagonal(similarity_test_test, -1)
        max_similarity_test = similarity_test_test.max(axis=1)

        max_similarity = np.maximum(max_similarity_train, max_similarity_test)

        return max_similarity, vectorizer_, train_vecs_


def _calculate_line_description_similarity(
    line_df: pd.DataFrame,
    vectorizer: skl_text.TfidfVectorizer | None = None,
    normal_vecs: sparse.spmatrix | None = None,
    is_train: bool = True,
) -> tuple[np.ndarray, skl_text.TfidfVectorizer | None, sparse.spmatrix | None]:
    normal_df = line_df[line_df["is_anomalous"] == 0]

    if is_train:
        vectorizer_ = skl_text.TfidfVectorizer()
        normal_vecs_ = vectorizer_.fit_transform(normal_df["line_description"])
        train_vecs = vectorizer_.transform(line_df["line_description"])

        similarity_matrix = pairwise.cosine_similarity(train_vecs, normal_vecs_)
        max_similarity = similarity_matrix.max(axis=1)

        return max_similarity, vectorizer_, normal_vecs_
    else:
        if vectorizer is None or normal_vecs is None:
            raise ValueError(
                "Must provide vectorizer and normal vectors when `is_train=False`."
            )

        test_vecs = vectorizer.transform(line_df["line_description"])

        similarity_matrix = pairwise.cosine_similarity(test_vecs, normal_vecs)
        max_similarity = similarity_matrix.max(axis=1)

        return max_similarity, None, None


def _extract_state(address: str) -> str | None:
    """Extract state abbreviation from a U.S. address."""
    match = re.search(r"\b([A-Z]{2})\s+\d{5}\b", address)
    return match.group(1) if match else None


def process_invoice(
    invoices: list[dict],
    is_train: bool = True,
    invoice_vectorizer: skl_text.TfidfVectorizer | None = None,
    invoice_train_vecs: sparse.spmatrix | None = None,
    duplicate_threshold: float = 0.95,
    line_vectorizer: skl_text.TfidfVectorizer | None = None,
    line_normal_vecs: sparse.spmatrix | None = None,
    phantom_threshold: float = 0.05,
    output_types: bool = False,
) -> tuple[
    pd.DataFrame,
    tuple[skl_text.TfidfVectorizer | None, sparse.spmatrix | None],
    tuple[skl_text.TfidfVectorizer | None, sparse.spmatrix | None],
]:
    """Process invoice with feature engineering for anomaly detection.

    Args:
        invoices (list[dict]): List of invoice JSON objects
        is_train (bool): Whether the invoices belong to the training set
            If True, new TF-IDF vectorizer and reference vectors will
            be constructed.
            If False, provided vectorizer and train vectors will be
            used for inference.
        invoice_vectorizer (Optional[TfidfVectorizer]): A pre-fitted
            vectorizer for full invoice text used in duplicate
            detection. Required when `is_train=False`.
        invoice_train_vecs (Optional[spmatrix]): TF-IDF matrix from
            training invoices for duplicate detection. Required when
            `is_train=False`.
        duplicate_threshold (float): Similarity threshold to flag
            duplicate invoices
        line_vectorizer (Optional[TfidfVectorizer]): A pre-fitted
            vectorizer for line item descriptions (used in phantom
            item detection). Required when `is_train=False`.
        line_normal_vecs (Optional[spmatrix]): TF-IDF matrix or sparse
            matrix for normal (non-anomalous) line item descriptions.
            Required when `is_train=False`.
        phantom_threshold (float): Similarity threshold to flag
            phantom items
        output_types (bool): Whether output anomaly types to support
            testing and model tuning

    Returns:
        tuple:
            - DataFrame: Engineered dataset with features
            - Optional[tuple[TfidfVectorizer, spmatrix]]:
                - Fitted vectorizer and training matrix for invoices
                (returned if `is_train=True`)
            - Optional[tuple[TfidfVectorizer, spmatrix]]:
                - Fitted vectorizer and reference matrix for line item
                descriptions (returned if `is_train=True`)

    Raises:
        ValueError: If `test=True` and some invoice entries are missing
            the `anomaly_types` field.
    """
    if not invoices:
        return pd.DataFrame(), (None, None), (None, None)

    if output_types and any("anomaly_types" not in invoice for invoice in invoices):
        raise ValueError(
            "`anomaly_types` not found in some invoices. "
            "If you're using real data, set `output_types=False`. "
            "Otherwise, regenerate the data with `output_types=True` "
            "to include anomaly labels."
        )

    # Calculate invoice similarities
    invoice_vectorizer_ = None
    invoice_train_vecs_ = None
    if is_train:
        max_invoice_similarity, invoice_vectorizer_, invoice_train_vecs_ = (
            _calculate_invoice_similarity(invoices)
        )
    else:
        max_invoice_similarity, _, _ = _calculate_invoice_similarity(
            invoices, invoice_vectorizer, invoice_train_vecs, is_train
        )

    records = []
    for i, (invoice, similarity) in enumerate(zip(invoices, max_invoice_similarity)):
        label = invoice.get("label", 0)
        extraction_fields = {
            entry["field"]: entry["value"] for entry in invoice.get("extractions", [])
        }
        anomaly_types = invoice.get("anomaly_types", [])

        # Grab and remove line_details from extraction_fields
        line_items = extraction_fields.pop("line_details", [])
        for item in line_items:
            record = {**extraction_fields, **item}
            record["invoice_idx"] = i
            record["invoice_similarity"] = similarity
            record["is_anomalous"] = label
            record["anomaly_types"] = anomaly_types
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
    df["expected_line_tax"] = df["expected_tax_rate"] * df["line_total"]
    df["line_tax_mismatch_flag"] = (
        df["expected_line_tax"] - df["line_tax"]
    ).abs() > 0.005  # Maximum deviation of rounding to 2 digits
    df["expected_tax"] = (
        df["expected_line_tax"].groupby(df["invoice_idx"]).transform("sum")
    )
    df["tax_mismatch_flag"] = (
        df["expected_tax"] - df["tax"]
    ).abs() > 0.005  # Maximum deviation of rounding to 2 digits

    # Line description similarity check
    line_vectorizer_ = None
    line_normal_vecs_ = None
    if is_train:
        max_line_similarity, line_vectorizer_, line_normal_vecs_ = (
            _calculate_line_description_similarity(df)
        )
    else:
        max_line_similarity, _, _ = _calculate_line_description_similarity(
            df, line_vectorizer, line_normal_vecs, is_train
        )
    df["line_description_similarity"] = max_line_similarity
    df["phantom_item_flag"] = df["line_description_similarity"] < phantom_threshold

    # Invoice similarity check
    df["duplicate_invoice_flag"] = df["invoice_similarity"] > duplicate_threshold

    # Payment days check
    df["payment_terms_numeric"] = df["payment_terms"].map(_PAYMENT_DAYS)
    df["invoice_age"] = (df["due_date"] - df["invoice_date"]).dt.days
    df["invoice_age_mismatch_flag"] = df["invoice_age"] != df["payment_terms_numeric"]

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
                # Output anomaly types for test support
                "anomaly_types": "first",
                # Anomaly label
                "is_anomalous": "max",
                # Original invoice data
                "merchant": "first",
                "invoice_date": "first",
                "merchant_branch": "first",
                "merchant_chain": "first",
                "due_date": "first",
                "payment_terms": "first",
                "grand_total": "first",
                "tax": "first",
                "po_number": "first",
                "country": "first",
                "currency": "first",
                "merchant_address": "first",
                "payment_method": "first",
                # Binary flags
                "invoice_age_mismatch_flag": "any",
                "line_tax_mismatch_flag": "any",
                "tax_mismatch_flag": "any",
                "phantom_item_flag": "any",
                "negative_qty_flag": "any",
                "duplicate_product_flag": "any",
                "merchant_mismatch_flag": "any",
                "duplicate_invoice_flag": "any",
                # Intermediate variables
                "line_total": "sum",
                "line_qty": "sum",
                "invoice_age": "first",
                "payment_terms_numeric": "first",
                "state": "first",
                "expected_tax_rate": "first",
                "expected_tax": "first",
                "line_description_similarity": "min",
                "invoice_similarity": "first",
            }
        )
        .reset_index()
        .drop(columns=["invoice_idx"])
    )

    if output_types:
        invoice_df = invoice_df.rename(
            columns={"anomaly_types": "_ANOMALY_TYPES_DROP_BEFORE_TRAINING_"}
        )
    else:
        invoice_df = invoice_df.drop("anomaly_types", axis=1)

    return (
        invoice_df,
        (invoice_vectorizer_, invoice_train_vecs_),
        (line_vectorizer_, line_normal_vecs_),
    )
