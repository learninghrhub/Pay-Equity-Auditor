import pandas as pd

def build_peer_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].astype(str).agg(" | ".join, axis=1)

def assign_peer_groups(df: pd.DataFrame, min_n: int = 20) -> pd.DataFrame:
    df = df.copy()

    df["peer_key"] = build_peer_key(df, ["grade", "job_family", "country"])
    counts = df["peer_key"].value_counts()
    small = df["peer_key"].map(counts) < min_n

    if small.any():
        df.loc[small, "peer_key"] = build_peer_key(df.loc[small], ["grade", "job_family"])
        counts2 = df["peer_key"].value_counts()
        small2 = df["peer_key"].map(counts2) < min_n

        if small2.any():
            df.loc[small2, "peer_key"] = build_peer_key(df.loc[small2], ["grade", "country"])
            counts3 = df["peer_key"].value_counts()
            small3 = df["peer_key"].map(counts3) < min_n

            if small3.any():
                df.loc[small3, "peer_key"] = build_peer_key(df.loc[small3], ["grade"])

    return df
import numpy as np
import pandas as pd

np.random.seed(42)

# -----------------------------
# Settings
# -----------------------------
N = 300  # change to 200/500 etc.
snapshot_date = "2026-02-01"

grades = ["A", "B", "C", "D", "E", "F"]
job_families = ["Finance", "HR", "IT", "Sales", "Operations", "Marketing"]
business_units = ["BU1", "BU2", "BU3"]
countries = ["Saudi Arabia", "UAE", "India"]

cities = {
    "Saudi Arabia": ["Jeddah", "Riyadh", "Dammam"],
    "UAE": ["Dubai", "Abu Dhabi", "Sharjah"],
    "India": ["Bengaluru", "Mumbai", "Delhi"],
}
cluster_prefix = {"Saudi Arabia": "KSA", "UAE": "UAE", "India": "IND"}

genders = ["Male", "Female"]
expat_local = ["Expat", "Local"]
nationalities = ["Saudi", "Emirati", "Indian", "Pakistani", "Egyptian", "Filipino", "British"]

employment_types = ["Permanent", "Contract"]

# Grade anchors (annual comp baseline) — dummy numbers for demo
grade_anchor = {"A": 60000, "B": 90000, "C": 130000, "D": 180000, "E": 240000, "F": 320000}
country_mult = {"Saudi Arabia": 1.20, "UAE": 1.35, "India": 0.55}
job_mult = {"Finance": 1.05, "HR": 0.95, "IT": 1.10, "Sales": 1.00, "Operations": 0.98, "Marketing": 0.97}


def pick_grade():
    return np.random.choice(grades, p=[0.10, 0.18, 0.22, 0.22, 0.18, 0.10])


rows = []
for i in range(N):
    employee_id = f"E{(i+1):05d}"

    grade = pick_grade()
    job_family = np.random.choice(job_families)
    business_unit = np.random.choice(business_units)

    country = np.random.choice(countries, p=[0.45, 0.25, 0.30])
    city = np.random.choice(cities[country])
    location_cluster = f"{cluster_prefix[country]}-{city}"

    gender = np.random.choice(genders, p=[0.65, 0.35])
    exp_loc = np.random.choice(expat_local, p=[0.40, 0.60])
    nationality = np.random.choice(nationalities)

    employment_type = np.random.choice(employment_types, p=[0.85, 0.15])
    fte = float(np.random.choice([1.0, 0.8, 0.6], p=[0.85, 0.10, 0.05]))

    # Tenure + time in grade
    tenure_years = float(np.clip(np.random.gamma(shape=2.0, scale=2.0), 0, 20))
    time_in_grade_years = float(np.round(np.random.uniform(0, min(tenure_years, 8)), 2)) if tenure_years > 0 else 0.0

    performance_rating = int(np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.50, 0.25, 0.05]))

    # Baseline base pay
    base = grade_anchor[grade] * country_mult[country] * job_mult[job_family]

    # Add tenure + performance effects
    base *= (1 + 0.015 * tenure_years) * (1 + 0.03 * (performance_rating - 3))

    # Random market noise
    base *= np.random.normal(1.0, 0.10)

    # Demo-only: slight gap effect (optional) for testing equity outputs
    if gender == "Female" and grade in ["D", "E", "F"]:
        base *= 0.97

    base_pay_annual = float(np.clip(base, 20000, 600000)) * fte

    # Allowances + bonus
    housing_allowance_annual = base_pay_annual * (0.18 if exp_loc == "Expat" else 0.10)
    transport_allowance_annual = base_pay_annual * (0.06 if country in ["Saudi Arabia", "UAE"] else 0.04)
    other_allowances_annual = base_pay_annual * float(np.random.uniform(0.01, 0.04))
    guaranteed_bonus_annual = base_pay_annual * float(np.random.uniform(0.00, 0.08))

    total_cash_annual = (
        base_pay_annual
        + housing_allowance_annual
        + transport_allowance_annual
        + other_allowances_annual
        + guaranteed_bonus_annual
    )

    # Flags
    critical_role_flag = int(np.random.rand() < 0.10)
    high_potential_flag = int(np.random.rand() < 0.08)

    # Optional org fields
    department = job_family
    cost_center = f"CC-{np.random.randint(100, 999)}"

    rows.append({
        "employee_id": employee_id,
        "snapshot_date": snapshot_date,

        "base_pay_annual": round(base_pay_annual, 2),
        "total_cash_annual": round(total_cash_annual, 2),

        "housing_allowance_annual": round(housing_allowance_annual, 2),
        "transport_allowance_annual": round(transport_allowance_annual, 2),
        "other_allowances_annual": round(other_allowances_annual, 2),
        "guaranteed_bonus_annual": round(guaranteed_bonus_annual, 2),

        "grade": grade,
        "job_family": job_family,
        "business_unit": business_unit,

        "country": country,
        "city": city,
        "location_cluster": location_cluster,

        "tenure_years": round(tenure_years, 2),
        "time_in_grade_years": round(time_in_grade_years, 2),
        "performance_rating": performance_rating,

        "critical_role_flag": critical_role_flag,
        "high_potential_flag": high_potential_flag,

        "fte": fte,
        "employment_type": employment_type,

        "gender": gender,
        "nationality": nationality,
        "expat_local": exp_loc,
    })

df = pd.DataFrame(rows)

out_path = "sample_template.csv"
df.to_csv(out_path, index=False)

print(f"✅ Created {out_path} with {len(df)} rows and {df.shape[1]} columns")
print("First 5 rows preview:")
print(df.head())