# Module 1 — Data Pipeline

## Zepto Data & AI Platform Capstone Project

This module implements the complete data pipeline for Task 1.

The pipeline follows this workflow:

Scrape → Clean → Convert → Store → Query → Validate

---

## 1. Objective

The objective of this module is to build a small data-engineering pipeline that:

1. Scrapes book data from `books.toscrape.com`
2. Cleans and converts the scraped fields
3. Converts GBP prices to INR using the required fixed project rate
4. Stores the data in a normalized SQLite database
5. Executes SQL queries for analysis
6. Reads SQL results into pandas DataFrames
7. Reproduces the JOIN result using `pandas.merge()`
8. Validates that the complete pipeline works successfully

The source website is a public scraping-practice website and does not require login credentials or an API key.

---

# 2. Data Source

Source:

`http://books.toscrape.com/`

The scraper collects book information including:

- Title
- Price in GBP
- Star rating
- Availability
- Category

The final dataset contains at least 60 books across multiple categories.

---

# 3. Project Structure

The `data_pipeline` directory contains the following files:

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