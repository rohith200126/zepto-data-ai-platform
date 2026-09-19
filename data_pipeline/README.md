# Module 1 — Data Pipeline

## Zepto Data & AI Platform Capstone Project

This module implements the complete data pipeline for Task 1.

The pipeline follows this workflow:

**Scrape → Clean → Convert → Store → Query → Validate**

---

## 1. Objective

The objective of this module is to build a data-engineering pipeline that:

1. Scrapes book data from `books.toscrape.com`
2. Cleans and converts the scraped fields
3. Converts GBP prices to INR using the required fixed project rate
4. Stores the data in a normalized SQLite database
5. Executes SQL queries for analysis
6. Reads SQL query results into pandas DataFrames
7. Reproduces the SQL JOIN result using `pandas.merge()`
8. Validates the complete pipeline

---

## 2. Data Source

Source:

http://books.toscrape.com/

The scraper collects:

- Title
- Price in GBP
- Star rating
- Availability
- Category

The final dataset contains at least 60 books across multiple categories.

The source website is a public scraping-practice website and does not require login credentials or an API key.

---

## 3. Project Structure

```text
data_pipeline/
│
├── scraper.py
├── database.py
├── README.md
├── books.csv
├── books.db
│
├── query_1_where_output.csv
├── query_2_order_limit_output.csv
├── query_3_distinct_output.csv
├── query_4_in_output.csv
├── query_5_between_output.csv
├── query_6_join_output.csv
│
├── pandas_merge_join_output.csv
├── sql_vs_pandas_join.csv
└── sql_queries.txt
4. Pipeline Workflow
books.toscrape.com
        |
        v
   Web Scraping
        |
        v
     books.csv
        |
        v
 Data Cleaning
        |
        v
 GBP → INR Conversion
        |
        v
   SQLite Database
        |
        +---- categories
        |
        +---- books
        |
        v
    SQL Queries
        |
        v
    pandas.read_sql()
        |
        v
    pandas.merge()
        |
        v
   Final Validation
5. Web Scraping

The scraping process uses:

requests
BeautifulSoup

The scraper collects:

title
price
star_rating
availability
category

The scraped data is saved into:

books.csv

The scraping process is automated and does not require manual copy-pasting.

6. Data Cleaning

The scraped data is cleaned before being inserted into the database.

Price

The original price contains the GBP currency symbol.

Example:

£51.77

It is converted into:

price_gbp = 51.77

The resulting column is a floating-point value.

Rating

The website provides ratings as text:

One
Two
Three
Four
Five

These are converted into integers:

One   → 1
Two   → 2
Three → 3
Four  → 4
Five  → 5
Availability

The availability information is converted into a boolean column:

in_stock
Parsing Errors

The pipeline handles parsing issues without allowing the complete process to crash.

Numeric parsing failures are handled using median imputation where applicable.

Rows that cannot be safely processed are dropped rather than leaving invalid values in the final dataset.

7. Currency Conversion

The project requires the following fixed conversion rate:

1 GBP = 105.50 INR

This is a project-defined fixed baseline rate.

It is not a live exchange rate and no external currency API is required.

The conversion formula is:

price_inr = price_gbp × 105.50
8. Database Design

SQLite is used as the relational database.

The database file is:

books.db

The database contains two normalized tables:

categories
books
Categories Table
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);
Books Table
CREATE TABLE books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER,
    FOREIGN KEY (category_id)
        REFERENCES categories(category_id)
);

The relationship is:

categories.category_id
          |
          v
books.category_id

This provides a normalized primary-key/foreign-key relationship between the two tables.

9. Database Loading

The cleaned and converted data is inserted into SQLite using Python's sqlite3 library.

The database pipeline:

Creates the database connection
Creates the categories table
Creates the books table
Clears previous records
Inserts unique categories
Creates a category lookup
Inserts all books
Commits the data
Validates the database contents

The database can be regenerated from the source data using the database pipeline script.

10. SQL Queries

The project requires SQL queries demonstrating:

SELECT
WHERE
ORDER BY
LIMIT
DISTINCT
IN
BETWEEN
JOIN

The implemented queries are stored in:

sql_queries.txt

The executed outputs are saved separately as CSV files.

Query 1 — WHERE

Demonstrates filtering records using the WHERE clause.

Output:

query_1_where_output.csv
Query 2 — ORDER BY and LIMIT

Demonstrates sorting and restricting the number of returned records.

Output:

query_2_order_limit_output.csv
Query 3 — DISTINCT

Demonstrates retrieving unique category values.

Output:

query_3_distinct_output.csv
Query 4 — IN

Demonstrates filtering records using the IN operator.

Output:

query_4_in_output.csv
Query 5 — BETWEEN

Demonstrates filtering numeric values using the BETWEEN operator.

Output:

query_5_between_output.csv
Query 6 — JOIN

Demonstrates a relational JOIN between the books and categories tables.

Output:

query_6_join_output.csv

The JOIN result is also reproduced independently using pandas.

11. Pandas Integration

SQL query results are read into pandas using:

pd.read_sql()

The resulting DataFrame contains:

book_id
title
price_gbp
price_inr
rating
in_stock
category_id
category_name
12. pandas.merge() Validation

The SQL JOIN result is independently reproduced using pandas.merge().

The process is:

Books DataFrame
       +
Categories DataFrame
       |
       v
  pandas.merge()
       |
       v
Merged DataFrame

The pandas merge output is saved to:

pandas_merge_join_output.csv

The SQL JOIN and pandas merge comparison is saved to:

sql_vs_pandas_join.csv

The validation confirms that the SQL JOIN result matches the pandas merge result.

13. Final Validation

The pipeline validates the required Task 1 criteria:

PASS: At least 60 books
PASS: At least 3 categories
PASS: price_gbp is float
PASS: rating is integer
PASS: in_stock is boolean
PASS: price_inr is float
PASS: Fixed GBP-INR rate used
PASS: SQL JOIN matches pandas.merge
14. Installation

The project requires the following Python libraries:

requests
beautifulsoup4
pandas

SQLite is provided through Python's standard library.

15. Running the Pipeline

The complete workflow is:

scraper.py
     |
     v
books.csv
     |
     v
database.py
     |
     v
books.db
     |
     +---- SQL query outputs
     |
     +---- pandas outputs
     |
     +---- validation

The pipeline runs end to end without manual data entry.

16. Generated Outputs

The following files are generated for verification:

books.csv
books.db

query_1_where_output.csv
query_2_order_limit_output.csv
query_3_distinct_output.csv
query_4_in_output.csv
query_5_between_output.csv
query_6_join_output.csv

pandas_merge_join_output.csv
sql_vs_pandas_join.csv

sql_queries.txt

17. Reproducibility

The database can be recreated from the scraped dataset using the provided Python scripts.

The required currency conversion always uses:

1 GBP = 105.50 INR

No external currency API is required for the graded conversion.

18. Task 1 Completion
============================================================
TASK 1 — DATA PIPELINE
============================================================

Scraping                         : COMPLETED
Data cleaning                    : COMPLETED
GBP price conversion             : COMPLETED
Fixed GBP → INR conversion       : COMPLETED
SQLite normalized schema         : COMPLETED
Categories table                 : COMPLETED
Books table                      : COMPLETED
SQL queries                      : COMPLETED
SQL JOIN                         : COMPLETED
pandas.read_sql                  : COMPLETED
pandas.merge                     : COMPLETED
SQL vs pandas validation         : COMPLETED
Final validation                 : PASSED

Fixed conversion rate            : 1 GBP = 105.50 INR

============================================================
TASK 1 DATABASE PIPELINE COMPLETED SUCCESSFULLY
============================================================