import collections
import copy
import datetime
import json
import random
from typing import Any

import faker
import numpy as np

# Set up the Faker generator with a fixed seed for reproducibility
fake = faker.Faker(["en_US"])

with open("synthetic_data_configs.json", "r") as f:
    synthetic_data_configs = json.load(f)


def _apply_line_anomaly(
    selected_products: list[dict],
    adjusted_unit_price: float,
    qty: int,
    line_total: float,
    anomaly_type: str | None = None,
) -> dict[str, Any] | None:
    line_anomaly_types = [
        "tax_calc",
        "line_total",
        "skip_tax",
        "duplicate_products",
        "negative_qty",
        "phantom_item",
    ]

    if anomaly_type is None:
        anomaly_type = random.choice(line_anomaly_types)

    assert anomaly_type in line_anomaly_types

    if anomaly_type == "tax_calc":
        line_tax = round(line_total * random.uniform(0.05, 0.25), 2)

        return {"line_tax": line_tax}

    if anomaly_type == "line_total":
        # Slightly off multiplication
        line_total = round(adjusted_unit_price * qty * random.uniform(0.5, 2.0), 2)

        return {"line_total": line_total}

    if anomaly_type == "skip_tax":
        line_tax = 0.0

        return {"line_tax": line_tax}

    if anomaly_type == "duplicate_products":
        duplicate_product = random.choice(selected_products)

        return {"duplicate_product": duplicate_product}

    if anomaly_type == "negative_qty":
        qty = -abs(qty)

        return {"qty": qty}

    if anomaly_type == "phantom_item":
        phantom_descriptions = [
            "Miscellaneous Supplies",
            "Service Fee",
            "Handling Charge",
            "Surcharge",
            "Administrative Fee",
            "Processing Fee",
            "Additional Service Charge",
        ]

        # Select a random phantom description
        phantom_description = random.choice(phantom_descriptions)

        # Add the phantom item to line details
        phantom_line_detail = {
            "line_count": 0,
            "line_description": phantom_description,
            "line_qty": str(random.randint(1, 10)),
            "line_tax": "{:.2f}".format(random.uniform(0.5, 10.0)),
            "line_total": "{:.2f}".format(random.uniform(5, 100)),
        }

        return {"phantom_line_detail": phantom_line_detail}

    return None


def _apply_general_anomaly(
    invoices: list[dict],
    merchant_name: str,
    merchant_type: str,
    tax_rate: float,
    invoice_date: datetime.datetime,
    due_date: datetime.datetime,
    due_days: int,
    anomaly_type: str | None = None,
) -> dict[str, Any] | None:
    general_anomaly_types = [
        "due_date_anomaly",
        "varied_wordings",
        "merchant_mismatch",
        "currency_anomaly",
        "state_tax_mismatch",
    ]

    if anomaly_type is None:
        anomaly_type = random.choice(general_anomaly_types)

    assert anomaly_type in general_anomaly_types

    if anomaly_type == "due_date_anomaly":
        due_date = invoice_date + datetime.timedelta(
            days=round(due_days * random.uniform(0.5, 1.5))
        )

        return {"due_date": due_date}

    if anomaly_type == "varied_wordings":
        merchant_invoices = [
            invoice
            for invoice in invoices
            if any(
                extraction["field"] == "merchant"
                and extraction["value"] == merchant_name
                for extraction in invoice["extractions"]
            )
        ]

        if merchant_invoices:
            base_invoice = random.choice(merchant_invoices)
            duplicate_invoice = _apply_variations(base_invoice)

            return {"duplicate_invoice": duplicate_invoice}

        return None

    if anomaly_type == "merchant_mismatch":
        chain_name = merchant_name_generator(
            [merchant_type], synthetic_data_configs["company_suffixes"]
        )[0]

        return {"chain_name": chain_name}

    if anomaly_type == "currency_anomaly":
        currency = random.choice(["CNY", "CAD", "JPY", "HKD", "GBP", "EUR"])

        return {"currency": currency}

    if anomaly_type == "state_tax_mismatch":
        while True:
            new_state = fake.state_abbr()

            if synthetic_data_configs["tax_rates"].get(new_state, 0.0) != tax_rate:
                break

        return {"state": new_state}

    return None


def _apply_variations(base_invoice: dict[str, list]) -> dict[str, list]:
    duplicate_invoice = copy.deepcopy(base_invoice)

    assert duplicate_invoice["extractions"][1]["field"] == "invoice_date"

    original_date = datetime.datetime.strptime(
        duplicate_invoice["extractions"][1]["value"], "%m/%d/%Y"
    )
    new_date = original_date + datetime.timedelta(days=random.randint(1, 30))
    duplicate_invoice["extractions"][1]["value"] = new_date.strftime("%m/%d/%Y")

    # Modify line item descriptions
    assert duplicate_invoice["extractions"][13]["field"] == "line_details"

    line_details = duplicate_invoice["extractions"][13]["value"]

    for line in line_details:
        if "line_description" not in line:  # Add key check
            continue

        base_product = next(
            (
                product_type
                for product_type in synthetic_data_configs["product_variations"]
                if line["line_description"].startswith(product_type)  # Key fix
            ),
            None,
        )

        if base_product:
            variation = random.choice(
                synthetic_data_configs["product_variations"][base_product]
            )
            line["line_description"] = (
                variation + " " + line["line_description"].split(" ", 1)[-1]
            )

    duplicate_invoice["extractions"][13]["value"] = line_details

    return duplicate_invoice


def merchant_name_generator(
    merchant_types: list[str], company_suffixes: list[str]
) -> tuple[str, str]:
    """Generate merchant names based on given merchant types.

    Args:
        merchant_types (list[str]): A list of possible merchant types
        company_suffixes (list[str]): A list of company suffixes

    Returns:
        tuple: A tuple where the first element is the generated
            merchant name, and the second element is the
            selected merchant type
    """
    name_prefix = random.choice(
        [
            fake.last_name(),
            fake.last_name() + " & " + fake.last_name(),
            fake.last_name() + "-" + fake.last_name(),
            fake.word().capitalize(),
            fake.word().capitalize() + fake.word().capitalize(),
        ]
    )

    merchant_type = random.choice(merchant_types)
    suffix = random.choice(company_suffixes)

    merchant_name = [name_prefix]
    if random.random() >= 0.7:
        merchant_name.append(merchant_type)
    if random.random() >= 0.5:
        merchant_name.append(suffix)

    return (" ".join(merchant_name), merchant_type)


def generate_merchant_data(num_merchants: int = 20) -> list[dict]:
    """Generate a diverse set of realistic merchant data.

    Args:
        num_merchants (int): Number of merchants to generate

    Returns:
        list: List of merchant dictionaries
    """
    merchant_types = synthetic_data_configs["merchant_types"]
    company_suffixes = synthetic_data_configs["company_suffixes"]

    merchants = [
        {
            "name": "Curtis Instruments, Inc.",
            "address": "200 Kisco Avenue Mount Kisco NY 10549 USA",
            "city": "Mount Kisco",
            "state": "NY",
            "country": "US",
            "type": "Electronics",
        }
    ]

    # Create a set to track unique merchant names
    existing_names = {merchants[0]["name"]}

    # Generate additional merchants
    for _ in range(num_merchants - 1):
        country_code = "US"

        fake.seed_instance(random.randint(1, 9999))

        # US address
        street = fake.street_address()
        city = fake.city()
        state = fake.state_abbr()
        zip_code = fake.zipcode()
        address = f"{street} {city} {state} {zip_code} USA"

        # Merchant Name Generator
        while True:
            merchant_name, merchant_type = merchant_name_generator(
                merchant_types, company_suffixes
            )

            # Check for uniqueness
            if merchant_name not in existing_names:
                existing_names.add(merchant_name)
                break

        merchants.append(
            {
                "name": merchant_name,
                "address": address,
                "city": city,
                "state": state,
                "country": country_code,
                "type": merchant_type,
            }
        )

    return merchants


def generate_product_catalog() -> dict[str, list]:
    """Generate a product catalog with categories and products.

    Returns:
        dict: Dictionary mapping categories to lists of products
    """
    categories = synthetic_data_configs["categories"]

    # Generate product catalog
    catalog = {}

    for category, product_types in categories.items():
        catalog[category] = []

        # Generate 15-30 products per category
        for _ in range(random.randint(15, 30)):
            product_type = random.choice(product_types)

            # Generate model number
            model_prefix = "".join(
                random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
                for _ in range(random.randint(1, 3))
            )
            model_suffix = "".join(
                random.choice("0123456789") for _ in range(random.randint(3, 5))
            )
            model = f"{model_prefix}-{model_suffix}"

            # Generate specification
            specs = [
                "Standard",
                "Premium",
                "Industrial",
                "Commercial",
                "Professional",
                "Heavy Duty",
                "Lightweight",
                "Compact",
                "Extended",
                "High Performance",
                "Economy",
                "Deluxe",
                "Ultimate",
                "Basic",
                "Advanced",
            ]
            spec = random.choice(specs)

            # Generate description
            description = f"{model} {spec} {product_type}"

            # Generate price based on category and specification
            base_price = {
                "Electronics": random.uniform(50, 500),
                "Industrial": random.uniform(100, 1000),
                "Office Supplies": random.uniform(10, 200),
                "Manufacturing": random.uniform(50, 800),
                "Automotive": random.uniform(30, 600),
                "Medical": random.uniform(100, 2000),
                "Construction": random.uniform(20, 500),
                "Technology": random.uniform(100, 3000),
                "Laboratory": random.uniform(50, 1500),
                "Engineering": random.uniform(100, 2000),
                "Aerospace": random.uniform(500, 5000),
                "Telecommunications": random.uniform(200, 2500),
            }[category]

            # Adjust price based on specification
            spec_multiplier = {
                "Standard": 1.0,
                "Premium": 1.5,
                "Industrial": 1.3,
                "Commercial": 1.2,
                "Professional": 1.4,
                "Heavy Duty": 1.3,
                "Lightweight": 0.9,
                "Compact": 0.85,
                "Extended": 1.25,
                "High Performance": 1.6,
                "Economy": 0.7,
                "Deluxe": 1.35,
                "Ultimate": 1.8,
                "Basic": 0.6,
                "Advanced": 1.5,
            }[spec]

            unit_price = round(base_price * spec_multiplier, 2)

            catalog[category].append(
                {
                    "description": description,
                    "unit_price": unit_price,
                    "model": model,
                    "spec": spec,
                    "type": product_type,
                }
            )

    return catalog


def generate_synthetic_invoice(
    num_invoices: int = 1,
    num_merchants: int = 10,
    general_anomaly_rate: float = 0.3,
    line_anomaly_rate: float = 0.05,
) -> list[dict]:
    """Generate synthetic invoice data based on the provided schema.

    Parameters:
        num_invoices (int): Number of synthetic invoices to generate
        num_invoices (int): Number of synthetic merchants to generate
        general_anomaly_rate (float): Rate at which to introduce general
            anomalies (anomaly per invoice)
        line_anomaly_rate (float): Rate at which to introduce line
            anomalies (anomaly per line item)

    Returns:
        list: List of invoice dictionaries following the
            extraction schema
    """
    invoices = []

    # Generate merchant data
    merchants = generate_merchant_data(num_merchants)

    # Generate product catalog
    product_catalog = generate_product_catalog()

    # Define payment terms with weighted probabilities
    payment_terms = [
        ("NET 30 DAYS", 0.4),
        ("NET 45 DAYS", 0.2),
        ("NET 60 DAYS", 0.1),
        ("DUE ON RECEIPT", 0.15),
        ("2/10 NET 30", 0.05),
        ("COD", 0.05),
        ("NET 15 DAYS", 0.05),
    ]

    # Define payment methods
    payment_methods = {
        "shipping": [
            "UPS Ground",
            "FedEx Express",
            "FedEx Ground",
            "USPS Priority",
            "DHL Express",
        ],
        "payment": [
            "Credit Card",
            "Wire Transfer",
            "ACH",
            "Check",
            "PayPal",
            "Net Banking",
        ],
    }

    # Define currencies with correct format
    currencies = {
        "US": "USD",
    }

    # Define tax rates by state
    tax_rates = synthetic_data_configs["tax_rates"]

    # Customer data generation
    customers = []
    num_customers = min(
        num_invoices // 5, 50
    )  # Create a reasonable number of customers
    for _ in range(max(1, num_customers)):  # Ensure at least one customer
        customer = {
            "name": fake.company(),
            "customer_id": f"CUST-{random.randint(10000, 99999)}",
            "address": fake.address().replace("\n", ", "),
            "contact_person": fake.name(),
            "email": fake.company_email(),
            "phone": fake.phone_number(),
        }
        customers.append(customer)

    # Create a distribution of invoices where some merchants and
    # customers appear more often
    # This mimics real world patterns where certain suppliers
    # are used more frequently
    merchant_weights = np.random.exponential(scale=1.0, size=len(merchants))
    merchant_weights = merchant_weights / np.sum(merchant_weights)

    customer_weights = np.random.exponential(scale=1.0, size=len(customers))
    customer_weights = customer_weights / np.sum(customer_weights)

    # Track merchant-customer relationships to create realistic patterns
    merchant_customers = collections.defaultdict(list)

    # Track invoice numbers by merchant
    # merchant_invoice_counters = {m["name"]: 1 for m in merchants}

    # Generate invoice data
    for _ in range(num_invoices):
        # Select merchant with weighted probability
        merchant = random.choices(merchants, weights=list(merchant_weights))[0]
        merchant_name = merchant["name"]
        merchant_address = merchant["address"]
        merchant_type = merchant["type"]
        # city = merchant["city"]
        state = merchant["state"]
        country = merchant["country"]
        currency = currencies[country]

        # Generate realistic invoice number
        """ invoice_prefix = "".join(random.choice(merchant_name.split()[0:1]).upper()[0:3])
        invoice_year = datetime.datetime.now().year
        invoice_num = merchant_invoice_counters[merchant_name]
        merchant_invoice_counters[merchant_name] += 1
        invoice_number = f"{invoice_prefix}{invoice_year}-{invoice_num:05d}" """

        # Generate invoice date (weighted toward recent dates)
        days_ago = int(np.random.exponential(scale=60))  # Mostly recent invoices
        days_ago = min(days_ago, 365)  # Cap at 1 year
        invoice_date = datetime.datetime.now() - datetime.timedelta(days=days_ago)

        # Format date according to country conventions
        if country in ["US", "CA"]:
            invoice_date_str = invoice_date.strftime("%m/%d/%Y")
        else:
            invoice_date_str = invoice_date.strftime("%d/%m/%Y")

        # Select payment term with weighted probability
        payment_term = random.choices(
            [term[0] for term in payment_terms],
            weights=[term[1] for term in payment_terms],
        )[0]

        # Calculate due date based on payment terms
        if "30" in payment_term:
            due_days = 60
        elif "45" in payment_term:
            due_days = 45
        elif "60" in payment_term:
            due_days = 60
        elif "15" in payment_term:
            due_days = 15
        elif "RECEIPT" in payment_term or "COD" in payment_term:
            due_days = 0
        else:  # Default for 2/10 NET 30 or other formats
            due_days = 30

        due_date = invoice_date + datetime.timedelta(days=due_days)

        # Format due date according to country conventions
        if country in ["US", "CA"]:
            due_date_str = due_date.strftime("%m/%d/%Y")
        else:
            due_date_str = due_date.strftime("%d/%m/%Y")

        # Select or create a customer relationship
        if merchant_name in merchant_customers and random.random() < 0.8:
            # 80% chance to use an existing customer for this merchant
            customer = random.choice(merchant_customers[merchant_name])
        else:
            # Either new merchant or 20% chance to add a new customer
            customer = np.random.choice(customers, p=customer_weights)
            merchant_customers[merchant_name].append(customer)

        # Generate PO number with several realistic formats
        po_formats = [
            f"{random.randint(10000, 99999)}-{random.randint(100, 999)}",
            f"PO-{random.randint(10000, 99999)}",
            f"{customer['customer_id']}-{random.randint(1000, 9999)}",
            f"{datetime.datetime.now().strftime('%y%m')}-{random.randint(1000, 9999)}",
            f"{random.randint(100000, 999999)}",
            f"{random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{random.randint(10000, 99999)}",
        ]
        po_number = random.choice(po_formats)

        # Determine payment method based on merchant type and
        # customer patterns
        if random.random() < 0.7:  # 70% chance it's a shipping method
            payment_method = random.choice(payment_methods["shipping"])
        else:  # 30% chance it's an actual payment method
            payment_method = random.choice(payment_methods["payment"])

        # Generate line items
        # line_items = []

        # Number of items follows a distribution centered on 3-4 items
        num_items_distribution = [0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02]
        num_items = np.random.choice(
            range(1, len(num_items_distribution) + 1), p=num_items_distribution
        )

        # For some merchants, force at least 2 line items to mimic
        # bundled purchases
        if random.random() < 0.3 and num_items == 1:
            num_items = 2

        # Select products from appropriate category for merchant type
        available_products = product_catalog.get(merchant_type, [])
        if not available_products:  # Fallback if category doesn't match
            available_products = random.choice(list(product_catalog.values()))

        # Select products without replacement if possible
        selected_products = random.sample(
            available_products, min(num_items, len(available_products))
        )

        # If we need more products than unique ones available,
        # allow repeats
        if num_items > len(selected_products):
            additional_products = [
                random.choice(available_products)
                for _ in range(num_items - len(selected_products))
            ]
            selected_products.extend(additional_products)

        # Calculate tax rates based on country and region
        tax_rate = tax_rates.get(state, 0.0)

        # Handle special case for shipping
        shipping_cost = 0
        if random.random() < 0.4:  # 40% chance of separate shipping charge
            shipping_cost = round(random.uniform(5, 50), 2)

        subtotal = 0
        total_tax = 0

        # Collect line details in the expected format
        formatted_line_details = []

        # Anomaly label
        label = 0

        # Generate each line item
        for i, product in enumerate(selected_products, 1):
            description = product["description"]
            unit_price = product["unit_price"]

            # Generate quantity with realistic distribution
            qty_distribution = {
                1: 0.3,  # 30% chance of qty 1
                2: 0.2,  # 20% chance of qty 2
                3: 0.1,  # 10% chance of qty 3
                4: 0.05,  # 5% chance of qty 4
                5: 0.05,  # 5% chance of qty 5
                10: 0.1,  # 10% chance of qty 10
                12: 0.05,  # 5% chance of qty 12
                24: 0.05,  # 5% chance of qty 24
                50: 0.05,  # 5% chance of qty 50
                100: 0.05,  # 5% chance of qty 100
            }

            qty = random.choices(
                list(qty_distribution.keys()), weights=list(qty_distribution.values())
            )[0]

            # Calculate line total with realistic price adjustments
            # Occasionally apply discounts for bulk purchases
            discount = 1.0
            if qty >= 10:
                discount = random.uniform(0.85, 0.95)
            elif qty >= 50:
                discount = random.uniform(0.75, 0.85)

            # Apply slight random variance to unit price
            adjusted_unit_price = unit_price * random.uniform(0.98, 1.02) * discount
            line_total = round(adjusted_unit_price * qty, 2)

            # Calculate line tax
            line_tax = round(line_total * tax_rate, 2) if tax_rate > 0 else 0.0

            # ----------------------------------------------------------
            # Introduce line anomalies
            if random.random() < line_anomaly_rate:
                anomaly = _apply_line_anomaly(
                    selected_products,
                    adjusted_unit_price,
                    qty,
                    line_total,
                )

                if anomaly is not None:
                    line_tax = anomaly.get("line_tax", line_tax)
                    line_total = anomaly.get("line_total", line_total)
                    qty = anomaly.get("qty", qty)

                    if "duplicate_product" in anomaly:
                        selected_products.append(anomaly["duplicate_product"])

                    if "phantom_line_detail" in anomaly:
                        phantom_line_detail = anomaly["phantom_line_detail"]
                        phantom_line_detail["line_count"] = str(
                            len(formatted_line_details) + 1
                        )
                        formatted_line_details.append(phantom_line_detail)

                    # Set anomaly label
                    label = 1
            # ----------------------------------------------------------

            # Add line details to our formatted output
            formatted_line_details.append(
                {
                    "line_count": str(i),
                    "line_description": description,
                    "line_qty": str(qty),
                    "line_tax": "{:.2f}".format(line_tax),
                    "line_total": "{:.2f}".format(line_total),
                    "model": product["model"],  # Add this line
                }
            )

            # Add to running totals
            subtotal += line_total
            total_tax += line_tax

        # Calculate grand total
        grand_total = subtotal + total_tax + shipping_cost

        # Introduce possible merchant mismatch anomaly
        chain_name = merchant_name

        # ----------------------------------------------------------
        # Introduce general anomalies
        if random.random() < general_anomaly_rate:
            anomaly = _apply_general_anomaly(
                invoices,
                merchant_name,
                merchant_type,
                tax_rate,
                invoice_date,
                due_date,
                due_days,
            )

            if anomaly is not None:
                due_date = anomaly.get("due_date", due_date)
                chain_name = anomaly.get("chain_name", chain_name)
                currency = anomaly.get("currency", currency)
                state = anomaly.get("state", state)

                if "due_date" in anomaly:
                    if country in ["US", "CA"]:
                        due_date_str = due_date.strftime("%m/%d/%Y")
                    else:
                        due_date_str = due_date.strftime("%d/%m/%Y")

                if "state" in anomaly:
                    splitted_address = merchant_address.split(" ")
                    splitted_address[-3] = state
                    merchant_address = " ".join(splitted_address)

                if "duplicate_invoice" in anomaly:
                    duplicate_invoice = anomaly["duplicate_invoice"]

                    # Set anomaly label for the duplicate invoice
                    duplicate_invoice["label"] = 1
                    invoices.append(duplicate_invoice)
                else:  # Set anomaly label for all other types of anomalies
                    label = 1
        # ----------------------------------------------------------

        # Create formatted invoice following the schema
        formatted_invoice = {
            "extractions": [
                {"field": "merchant", "value": merchant_name},
                {"field": "invoice_date", "value": invoice_date_str},
                {"field": "merchant_branch", "value": merchant_name},
                {"field": "merchant_chain", "value": chain_name},
                {"field": "due_date", "value": due_date_str},
                {"field": "payment_terms", "value": payment_term},
                {"field": "grand_total", "value": "{:.2f}".format(grand_total)},
                {"field": "tax", "value": "{:.2f}".format(total_tax)},
                {"field": "po_number", "value": po_number},
                {
                    "field": "merchant_address",
                    "value": merchant_address.replace(",", ""),
                },
                {"field": "payment_method", "value": payment_method},
                {"field": "country", "value": country},
                {"field": "currency", "value": currency},
                {"field": "line_details", "value": formatted_line_details},
            ],
            "label": label,
        }

        invoices.append(formatted_invoice)

    return invoices


def analyze_synthetic_data(invoices: list[dict]) -> None:
    """Analyze the generated synthetic data.

    Args:
        invoices (list): List of invoice dictionaries
    """
    # Count merchants
    merchants = set()
    countries = set()
    total_value = 0
    line_items_count = 0

    for invoice in invoices:
        extractions = {item["field"]: item["value"] for item in invoice["extractions"]}

        if "merchant" in extractions:
            merchants.add(extractions["merchant"])

        if "country" in extractions:
            countries.add(extractions["country"])

        if "grand_total" in extractions:
            try:
                total_value += float(extractions["grand_total"])
            except ValueError:
                pass

        if "line_details" in extractions:
            line_items = extractions["line_details"]
            if isinstance(line_items, list):
                line_items_count += len(line_items)

    print(f"Analysis of {len(invoices)} invoices:")
    print(f"- Unique merchants: {len(merchants)}")
    print(f"- Total line items: {line_items_count}")
    print(f"- Average line items per invoice: {line_items_count/len(invoices):.2f}")

    # Count frequency of fields
    field_counts = collections.defaultdict(int)
    for invoice in invoices:
        for item in invoice["extractions"]:
            field_counts[item["field"]] += 1

    print("\nField frequency:")
    for field, count in sorted(field_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(invoices) * 100
        print(f"- {field}: {count} ({percentage:.1f}%)")


def save_synthetic_data(
    invoices: list[dict], base_filename: str = "synthetic_invoices"
) -> None:
    """Save synthetic invoices to JSON files.

    Args:
        invoices (list): List of invoice dictionaries
        base_filename (str): Base filename without extension
    """
    # Save all invoices to one file
    all_filename = f"{base_filename}.json"
    with open(all_filename, "w") as f:
        json.dump(invoices, f, indent=2)

    print(f"Generated {len(invoices)} synthetic invoices and saved to {all_filename}")

    # Save in JSONL format as well (one invoice per line,
    # no indentation)
    jsonl_filename = f"{base_filename}.jsonl"
    with open(jsonl_filename, "w") as f:
        for invoice in invoices:
            f.write(json.dumps(invoice) + "\n")

    print(f"Saved invoices in JSONL format to {jsonl_filename}")


def generate_dataset(
    *,
    num_invoices: int = 100,
    num_merchants: int = 1000,
    general_anomaly_rate: float = 0.3,
    line_anomaly_rate: float = 0.05,
    seed: int = 42,
) -> list[dict]:
    """Generate a complete synthetic invoice dataset.

    Args:
        num_invoices (int): Number of invoices to generate
        num_merchants (int): Number of merchants to generate
        general_anomaly_rate (float): Rate of general anomalies
            (anomaly per invoice) to introduce
        line_anomaly_rate (float): Rate of line anomalies
            (anomaly per line item) to introduce
        seed (int): Random seed for reproducibility

    Returns:
        list: Generated invoices
    """
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    faker.Faker.seed(seed)

    print(
        f"Generating {num_invoices} synthetic invoices across a set of "
        f"{num_merchants} merchants (general anomaly rate: {general_anomaly_rate:.1%}, "
        f"line anomaly rate: {line_anomaly_rate:.1%})..."
    )
    invoices = generate_synthetic_invoice(
        num_invoices, num_merchants, general_anomaly_rate, line_anomaly_rate
    )

    # Analyze the generated data
    analyze_synthetic_data(invoices)

    # Save the data
    save_synthetic_data(invoices)

    return invoices


if __name__ == "__main__":
    generate_dataset(
        num_invoices=10000,
        num_merchants=1000,
        general_anomaly_rate=0.3,
        line_anomaly_rate=0.05,
    )
