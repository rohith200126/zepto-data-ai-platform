import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import csv
import pandas as pd

url = "https://books.toscrape.com/"

current_url = url

total_books = 0
books_data = []

while total_books < 60:

    # Open the current page
    response = requests.get(current_url)
    response.encoding = "utf-8"

    # Convert HTML into BeautifulSoup object
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all books on the current page
    books = soup.find_all("article", class_="product_pod")

    print("Current page:", current_url)
    print("Books on this page:", len(books))

    # Extract information from each book
    for book in books:

        # -------------------------
        # Title
        # -------------------------
        title_tag = book.find("h3")
        title = title_tag.find("a")["title"]

        # -------------------------
        # Price
        # -------------------------
        price_tag = book.find("p", class_="price_color")
        price = price_tag.text.strip()

        # Remove £ symbol and convert to float
        price_gbp = float(price.replace("£", ""))

        # -------------------------
        # Rating
        # -------------------------
        rating_tag = book.find("p", class_="star-rating")
        rating_text = rating_tag["class"][1]

        rating_map = {
            "One": 1,
            "Two": 2,
            "Three": 3,
            "Four": 4,
            "Five": 5
        }

        rating = rating_map[rating_text]

        # -------------------------
        # Availability
        # -------------------------
        availability_tag = book.find(
            "p",
            class_="instock availability"
        )

        availability = availability_tag.get_text(
            strip=True
        )

        # Convert availability to Boolean
        availability = "In stock" in availability

        # -------------------------
        # Book link
        # -------------------------
        book_link = title_tag.find("a")["href"]
        book_url = urljoin(current_url, book_link)

        # Open individual book page
        book_response = requests.get(book_url)
        book_response.encoding = "utf-8"

        book_soup = BeautifulSoup(
            book_response.text,
            "html.parser"
        )

        # -------------------------
        # Category
        # -------------------------
        breadcrumb = book_soup.find(
            "ul",
            class_="breadcrumb"
        )

        category = breadcrumb.find_all("li")[2].get_text(
            strip=True
        )

        # -------------------------
        # Create dictionary
        # -------------------------
        book_data = {
            "title": title,
            "price_gbp": price_gbp,
            "rating": rating,
            "availability": availability,
            "category": category
        }

        # Add dictionary to list
        books_data.append(book_data)

        # -------------------------
        # Display information
        # -------------------------
        print("Title:", title)
        print("Price GBP:", price_gbp)
        print("Rating:", rating)
        print("Availability:", availability)
        print("Category:", category)
        print()

        total_books += 1

        # Stop after 60 books
        if total_books >= 60:
            break

    # Stop pagination after 60 books
    if total_books >= 60:
        break

    # Find Next button
    next_button = soup.find("li", class_="next")

    # Get next page link
    next_link = next_button.find("a")["href"]

    # Create complete URL
    current_url = urljoin(current_url, next_link)


# -------------------------
# Final counts
# -------------------------
print("Total books scraped:", total_books)
print("Total records stored:", len(books_data))


# -------------------------
# Save data to CSV
# -------------------------
with open(
    "books.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "title",
            "price_gbp",
            "rating",
            "availability",
            "category"
        ]
    )

    writer.writeheader()
    writer.writerows(books_data)

print("Data saved to books.csv")


# -------------------------
# Validate CSV
# -------------------------
df = pd.read_csv("books.csv")

print("CSV shape:", df.shape)

print("Column names:")
print(df.columns.tolist())

print("Missing values:")
print(df.isnull().sum())

print("Data types:")
print(df.dtypes)