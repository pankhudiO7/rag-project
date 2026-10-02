import arxiv
import os

# Define your categories and download directory
categories = ["cs.LG", "stat.ML","cs.AI","cs.CL","cs.RO", "cs.CR"] # Example categories: Computer Science - Machine Learning, Statistics - Machine Learning
download_dir = "./arxiv_papers"

# Create the download directory if it doesn't exist
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Build the query string for multiple categories (using OR logic in arXiv API)
query = "cat:" + " OR cat:".join(categories)

search = arxiv.Search(
    query = query,
    max_results = 300,
    sort_by = arxiv.SortCriterion.SubmittedDate, # Correct attribute
    sort_order = arxiv.SortOrder.Descending
)

# Download the papers
for result in search.results():
    print(f"Downloading {result.entry_id}...")
    try:
        # Download the PDF using the result's helper method
        result.download_pdf(dirpath=download_dir)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading {result.entry_id}: {e}")
