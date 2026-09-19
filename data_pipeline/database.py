import sqlite3
import pandas as pd
import os


# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = "books.csv"
DATABASE_FILE = "books.db"

# Project-defined fixed conversion rate
GBP_TO_INR = 105.50


# ============================================================
# 1. LOAD SCRAPED CSV DATA
# ============================================================

print("=" * 70)
print("TASK 1 - DATA PIPELINE")
print("=" * 70)

print("\n1. Loading CSV data...")

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(
        f"{CSV_FILE} not found. Run scraper.py first."
    )

df = pd.read_csv(CSV_FILE)

print("CSV data loaded successfully!")
print("Total records in CSV:", len(df))


# ============================================================
# 2. CHECK REQUIRED SCRAPED COLUMNS
# ============================================================

print("\n2. Checking required columns...")

required_columns = [
    "title",
    "price_gbp",
    "rating",
    "availability",
    "category"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns in CSV: {missing_columns}"
    )

print("All required columns are present.")


# ============================================================
# 3. CLEAN DATA
# ============================================================

print("\n3. Cleaning data...")

# ------------------------------------------------------------
# Title
# ------------------------------------------------------------

df["title"] = df["title"].astype(str).str.strip()


# ------------------------------------------------------------
# Price GBP
# ------------------------------------------------------------

df["price_gbp"] = pd.to_numeric(
    df["price_gbp"],
    errors="coerce"
)

price_missing = df["price_gbp"].isna().sum()

if price_missing > 0:
    median_price = df["price_gbp"].median()

    df["price_gbp"] = df["price_gbp"].fillna(
        median_price
    )

    print(
        f"Price parsing failures: {price_missing}"
    )
    print(
        f"Missing prices replaced with median: {median_price}"
    )
else:
    print("Price parsing successful for all rows.")


# ------------------------------------------------------------
# Rating
# ------------------------------------------------------------

rating_mapping = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

# Handle both text ratings and already numeric ratings
if df["rating"].dtype == object:

    df["rating"] = (
        df["rating"]
        .astype(str)
        .str.strip()
        .map(rating_mapping)
    )

else:

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )


rating_missing = df["rating"].isna().sum()

if rating_missing > 0:

    median_rating = round(
        df["rating"].median()
    )

    df["rating"] = df["rating"].fillna(
        median_rating
    )

    print(
        f"Rating parsing failures: {rating_missing}"
    )

    print(
        f"Missing ratings replaced with median: {median_rating}"
    )

else:

    print("Rating parsing successful for all rows.")


df["rating"] = df["rating"].astype(int)


# ------------------------------------------------------------
# Availability -> in_stock
# ------------------------------------------------------------

def parse_availability(value):

    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if "in stock" in value:
        return True

    if "out of stock" in value:
        return False

    if value in ["true", "1"]:
        return True

    if value in ["false", "0"]:
        return False

    return None


df["in_stock"] = df["availability"].apply(
    parse_availability
)


availability_missing = df["in_stock"].isna().sum()

if availability_missing > 0:

    print(
        f"Availability parsing failures: "
        f"{availability_missing}"
    )

    # Since in_stock is boolean and not numeric,
    # use the most common valid value.
    mode_value = df["in_stock"].mode()

    if len(mode_value) > 0:
        replacement = mode_value.iloc[0]
    else:
        replacement = True

    df["in_stock"] = df["in_stock"].fillna(
        replacement
    )

    print(
        f"Invalid availability values replaced with: "
        f"{replacement}"
    )

else:

    print(
        "Availability parsing successful for all rows."
    )


df["in_stock"] = df["in_stock"].astype(bool)


# ------------------------------------------------------------
# Category
# ------------------------------------------------------------

df["category"] = (
    df["category"]
    .astype(str)
    .str.strip()
)


# ============================================================
# 4. REMOVE INVALID ROWS
# ============================================================

print("\n4. Checking for invalid rows...")

before_rows = len(df)

df = df.dropna(
    subset=[
        "title",
        "price_gbp",
        "rating",
        "in_stock",
        "category"
    ]
)

after_rows = len(df)

dropped_rows = before_rows - after_rows

if dropped_rows > 0:

    print(
        f"Dropped {dropped_rows} rows "
        f"because required fields were invalid."
    )

else:

    print("No rows needed to be dropped.")


# ============================================================
# 5. CONVERT GBP TO INR
# ============================================================

print("\n5. Converting GBP to INR...")

df["price_inr"] = (
    df["price_gbp"] * GBP_TO_INR
)

print(
    f"Fixed project conversion rate: "
    f"1 GBP = {GBP_TO_INR} INR"
)


# ============================================================
# 6. FINAL CLEAN DATAFRAME
# ============================================================

cleaned_df = df[
    [
        "title",
        "price_gbp",
        "rating",
        "in_stock",
        "category",
        "price_inr"
    ]
].copy()


# Make sure data types are correct
cleaned_df["title"] = cleaned_df["title"].astype(str)

cleaned_df["price_gbp"] = cleaned_df[
    "price_gbp"
].astype(float)

cleaned_df["rating"] = cleaned_df[
    "rating"
].astype(int)

cleaned_df["in_stock"] = cleaned_df[
    "in_stock"
].astype(bool)

cleaned_df["category"] = cleaned_df[
    "category"
].astype(str)

cleaned_df["price_inr"] = cleaned_df[
    "price_inr"
].astype(float)


print("\nCleaned DataFrame:")
print(cleaned_df.head())


print("\nData types:")
print(cleaned_df.dtypes)


print("\nCleaned DataFrame shape:")
print(cleaned_df.shape)


# ============================================================
# 7. CREATE FRESH DATABASE
# ============================================================

print("\n6. Creating SQLite database...")


# Remove previous database so every run starts fresh
if os.path.exists(DATABASE_FILE):

    os.remove(DATABASE_FILE)

    print("Old database removed.")


connection = sqlite3.connect(
    DATABASE_FILE
)

cursor = connection.cursor()


# Enable foreign key enforcement
cursor.execute(
    "PRAGMA foreign_keys = ON"
)

print("Database connection created!")


# ============================================================
# 8. CREATE NORMALIZED CATEGORIES TABLE
# ============================================================

cursor.execute("""
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
)
""")


# ============================================================
# 9. CREATE NORMALIZED BOOKS TABLE
# ============================================================

cursor.execute("""
CREATE TABLE books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    price_inr REAL NOT NULL,
    rating INTEGER NOT NULL,
    in_stock INTEGER NOT NULL,
    category_id INTEGER NOT NULL,

    FOREIGN KEY (category_id)
        REFERENCES categories(category_id)
)
""")


connection.commit()

print("Normalized tables created!")


# ============================================================
# 10. INSERT UNIQUE CATEGORIES
# ============================================================

print("\n7. Inserting categories...")

unique_categories = (
    cleaned_df["category"]
    .dropna()
    .unique()
)


for category in unique_categories:

    cursor.execute(
        """
        INSERT INTO categories (
            category_name
        )
        VALUES (?)
        """,
        (category,)
    )


connection.commit()

print(
    "Categories inserted:",
    len(unique_categories)
)


# ============================================================
# 11. CREATE CATEGORY LOOKUP
# ============================================================

cursor.execute("""
SELECT
    category_id,
    category_name
FROM categories
""")


category_rows = cursor.fetchall()


category_map = {
    category_name: category_id
    for category_id, category_name
    in category_rows
}


# ============================================================
# 12. INSERT BOOKS
# ============================================================

print("\n8. Inserting books...")

for _, row in cleaned_df.iterrows():

    category_id = category_map[
        row["category"]
    ]

    cursor.execute(
        """
        INSERT INTO books (
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            row["title"],
            row["price_gbp"],
            row["price_inr"],
            row["rating"],
            int(row["in_stock"]),
            category_id
        )
    )


connection.commit()


print(
    "Books inserted:",
    len(cleaned_df)
)


# ============================================================
# 13. VERIFY DATABASE STRUCTURE
# ============================================================

print("\n9. Categories table structure:")

cursor.execute("""
PRAGMA table_info(categories)
""")

for row in cursor.fetchall():

    print(row)


print("\nBooks table structure:")

cursor.execute("""
PRAGMA table_info(books)
""")

for row in cursor.fetchall():

    print(row)


# ============================================================
# 14. SQL QUERY 1
# SELECT + WHERE
# ============================================================

query_1 = """
SELECT
    title,
    price_gbp,
    rating,
    in_stock
FROM books
WHERE rating >= 4
"""


print("\n" + "=" * 70)
print("SQL QUERY 1 - SELECT + WHERE")
print("=" * 70)

print(query_1)

result_1 = pd.read_sql(
    query_1,
    connection
)

print(result_1.to_string(index=False))


# ============================================================
# 15. SQL QUERY 2
# ORDER BY + LIMIT
# ============================================================

query_2 = """
SELECT
    title,
    price_gbp,
    price_inr
FROM books
ORDER BY price_gbp DESC
LIMIT 10
"""


print("\n" + "=" * 70)
print("SQL QUERY 2 - ORDER BY + LIMIT")
print("=" * 70)

print(query_2)

result_2 = pd.read_sql(
    query_2,
    connection
)

print(result_2.to_string(index=False))


# ============================================================
# 16. SQL QUERY 3
# DISTINCT
# ============================================================

query_3 = """
SELECT DISTINCT
    category_name
FROM categories
ORDER BY category_name
"""


print("\n" + "=" * 70)
print("SQL QUERY 3 - DISTINCT")
print("=" * 70)

print(query_3)

result_3 = pd.read_sql(
    query_3,
    connection
)

print(result_3.to_string(index=False))


# ============================================================
# 17. SQL QUERY 4
# IN
# ============================================================

query_4 = """
SELECT
    title,
    rating,
    price_gbp
FROM books
WHERE rating IN (4, 5)
ORDER BY rating DESC, price_gbp DESC
"""


print("\n" + "=" * 70)
print("SQL QUERY 4 - IN")
print("=" * 70)

print(query_4)

result_4 = pd.read_sql(
    query_4,
    connection
)

print(result_4.to_string(index=False))


# ============================================================
# 18. SQL QUERY 5
# BETWEEN
# ============================================================

query_5 = """
SELECT
    title,
    price_gbp,
    rating
FROM books
WHERE price_gbp BETWEEN 20 AND 40
ORDER BY price_gbp
"""


print("\n" + "=" * 70)
print("SQL QUERY 5 - BETWEEN")
print("=" * 70)

print(query_5)

result_5 = pd.read_sql(
    query_5,
    connection
)

print(result_5.to_string(index=False))


# ============================================================
# 19. SQL QUERY 6
# JOIN
# ============================================================

query_6 = """
SELECT
    books.book_id,
    books.title,
    books.price_gbp,
    books.price_inr,
    books.rating,
    books.in_stock,
    books.category_id,
    categories.category_name
FROM books
JOIN categories
    ON books.category_id = categories.category_id
ORDER BY books.rating DESC, books.price_gbp DESC
LIMIT 10
"""


print("\n" + "=" * 70)
print("SQL QUERY 6 - JOIN")
print("=" * 70)

print(query_6)

join_sql_df = pd.read_sql(
    query_6,
    connection
)

print(
    join_sql_df.to_string(index=False)
)


# ============================================================
# 20. SAVE SQL QUERY OUTPUTS
# ============================================================

print("\n10. Saving SQL query outputs...")

result_1.to_csv(
    "query_1_where_output.csv",
    index=False
)

result_2.to_csv(
    "query_2_order_limit_output.csv",
    index=False
)

result_3.to_csv(
    "query_3_distinct_output.csv",
    index=False
)

result_4.to_csv(
    "query_4_in_output.csv",
    index=False
)

result_5.to_csv(
    "query_5_between_output.csv",
    index=False
)

join_sql_df.to_csv(
    "query_6_join_output.csv",
    index=False
)

print("All query outputs saved.")


# ============================================================
# 21. SAVE QUERY STRINGS
# ============================================================

queries = {
    "QUERY 1 - SELECT WHERE": query_1,
    "QUERY 2 - ORDER BY LIMIT": query_2,
    "QUERY 3 - DISTINCT": query_3,
    "QUERY 4 - IN": query_4,
    "QUERY 5 - BETWEEN": query_5,
    "QUERY 6 - JOIN": query_6
}


with open(
    "sql_queries.txt",
    "w",
    encoding="utf-8"
) as file:

    for query_name, query_text in queries.items():

        file.write("=" * 70 + "\n")

        file.write(
            query_name + "\n"
        )

        file.write("=" * 70 + "\n")

        file.write(
            query_text.strip() + "\n\n"
        )


print("SQL query strings saved to sql_queries.txt")


# ============================================================
# 22. READ TWO QUERY RESULTS USING pd.read_sql()
# ============================================================

print("\n" + "=" * 70)
print("PANDAS read_sql() VERIFICATION")
print("=" * 70)


pandas_query_1 = pd.read_sql(
    query_1,
    connection
)

pandas_query_2 = pd.read_sql(
    query_2,
    connection
)


print("\nQuery 1 loaded using pd.read_sql():")
print(
    pandas_query_1.head().to_string(
        index=False
    )
)


print("\nQuery 2 loaded using pd.read_sql():")
print(
    pandas_query_2.head().to_string(
        index=False
    )
)


# ============================================================
# 23. CREATE IN-MEMORY PANDAS DATAFRAMES
# ============================================================

print("\n" + "=" * 70)
print("PANDAS DATAFRAMES FOR pd.merge()")
print("=" * 70)


books_memory_df = pd.read_sql(
    """
    SELECT
        book_id,
        title,
        price_gbp,
        price_inr,
        rating,
        in_stock,
        category_id
    FROM books
    """,
    connection
)


categories_memory_df = pd.read_sql(
    """
    SELECT
        category_id,
        category_name
    FROM categories
    """,
    connection
)


print("\nBooks DataFrame:")
print(
    books_memory_df.head().to_string(
        index=False
    )
)


print("\nCategories DataFrame:")
print(
    categories_memory_df.head().to_string(
        index=False
    )
)


# ============================================================
# 24. REPRODUCE JOIN USING pd.merge()
# ============================================================

print("\n" + "=" * 70)
print("PANDAS MERGE - REPRODUCING SQL JOIN")
print("=" * 70)


merged_df = pd.merge(
    books_memory_df,
    categories_memory_df,
    on="category_id",
    how="inner"
)


# Same ordering and LIMIT as SQL JOIN
merged_df = (
    merged_df
    .sort_values(
        by=["rating", "price_gbp"],
        ascending=[False, False]
    )
    .head(10)
)


# Reorder columns to match SQL JOIN output
merged_df = merged_df[
    [
        "book_id",
        "title",
        "price_gbp",
        "price_inr",
        "rating",
        "in_stock",
        "category_id",
        "category_name"
    ]
]


print("\nPandas merge result:")
print(
    merged_df.to_string(
        index=False
    )
)


# ============================================================
# 25. COMPARE SQL JOIN AND PANDAS MERGE
# ============================================================

print("\n" + "=" * 70)
print("SQL JOIN vs PANDAS MERGE")
print("=" * 70)


sql_join_compare = join_sql_df.reset_index(
    drop=True
)

pandas_merge_compare = merged_df.reset_index(
    drop=True
)


# Ensure identical data types before comparison
sql_join_compare["book_id"] = (
    sql_join_compare["book_id"].astype(int)
)

pandas_merge_compare["book_id"] = (
    pandas_merge_compare["book_id"].astype(int)
)


sql_join_compare["category_id"] = (
    sql_join_compare["category_id"].astype(int)
)

pandas_merge_compare["category_id"] = (
    pandas_merge_compare["category_id"].astype(int)
)


sql_join_compare["rating"] = (
    sql_join_compare["rating"].astype(int)
)

pandas_merge_compare["rating"] = (
    pandas_merge_compare["rating"].astype(int)
)


sql_join_compare["in_stock"] = (
    sql_join_compare["in_stock"].astype(int)
)

pandas_merge_compare["in_stock"] = (
    pandas_merge_compare["in_stock"].astype(int)
)


# Round floating-point values
sql_join_compare["price_gbp"] = (
    sql_join_compare["price_gbp"].round(6)
)

pandas_merge_compare["price_gbp"] = (
    pandas_merge_compare["price_gbp"].round(6)
)


sql_join_compare["price_inr"] = (
    sql_join_compare["price_inr"].round(6)
)

pandas_merge_compare["price_inr"] = (
    pandas_merge_compare["price_inr"].round(6)
)


join_matches = sql_join_compare.equals(
    pandas_merge_compare
)


print(
    "\nDo SQL JOIN and pandas.merge() "
    "produce equivalent results?"
)

print(join_matches)


# ============================================================
# 26. SAVE PANDAS MERGE OUTPUT
# ============================================================

merged_df.to_csv(
    "pandas_merge_join_output.csv",
    index=False
)


# ============================================================
# 27. SAVE SIDE-BY-SIDE COMPARISON
# ============================================================

side_by_side = pd.concat(
    [
        sql_join_compare.add_prefix(
            "sql_"
        ),
        pandas_merge_compare.add_prefix(
            "pandas_"
        )
    ],
    axis=1
)


side_by_side.to_csv(
    "sql_vs_pandas_join.csv",
    index=False
)


print(
    "\nSQL vs pandas comparison saved to:"
)

print(
    "sql_vs_pandas_join.csv"
)


# ============================================================
# 28. FINAL DATABASE COUNTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)


cursor.execute(
    "SELECT COUNT(*) FROM books"
)

total_books = cursor.fetchone()[0]


cursor.execute(
    "SELECT COUNT(*) FROM categories"
)

total_categories = cursor.fetchone()[0]


print(
    "Total books in database:",
    total_books
)

print(
    "Total categories in database:",
    total_categories
)


# ============================================================
# 29. VERIFY MINIMUM CATEGORY REQUIREMENT
# ============================================================

if total_categories >= 3:

    print(
        "Category requirement: PASSED"
    )

else:

    print(
        "Category requirement: FAILED"
    )


# ============================================================
# 30. VERIFY MINIMUM BOOK REQUIREMENT
# ============================================================

if total_books >= 60:

    print(
        "Book count requirement: PASSED"
    )

else:

    print(
        "Book count requirement: FAILED"
    )


# ============================================================
# 31. VERIFY REQUIRED DATA TYPES
# ============================================================

print("\nFinal DataFrame data types:")

print(
    cleaned_df[
        [
            "price_gbp",
            "rating",
            "in_stock",
            "price_inr"
        ]
    ].dtypes
)


# ============================================================
# 32. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("TASK 1 VALIDATION")
print("=" * 70)


checks = {

    "At least 60 books":
        total_books >= 60,

    "At least 3 categories":
        total_categories >= 3,

    "price_gbp is float":
        pd.api.types.is_float_dtype(
            cleaned_df["price_gbp"]
        ),

    "rating is integer":
        pd.api.types.is_integer_dtype(
            cleaned_df["rating"]
        ),

    "in_stock is boolean":
        pd.api.types.is_bool_dtype(
            cleaned_df["in_stock"]
        ),

    "price_inr is float":
        pd.api.types.is_float_dtype(
            cleaned_df["price_inr"]
        ),

    "Fixed GBP-INR rate used":
        GBP_TO_INR == 105.50,

    "SQL JOIN matches pandas.merge":
        join_matches
}


all_passed = True


for check_name, result in checks.items():

    status = "PASS" if result else "FAIL"

    print(
        f"{status}: {check_name}"
    )

    if not result:
        all_passed = False


# ============================================================
# 33. CLOSE DATABASE
# ============================================================

connection.close()


# ============================================================
# 34. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)

if all_passed:

    print(
        "TASK 1 DATABASE PIPELINE COMPLETED SUCCESSFULLY!"
    )

else:

    print(
        "TASK 1 COMPLETED WITH SOME FAILED CHECKS."
    )

print("=" * 70)