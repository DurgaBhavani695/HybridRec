import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.makedirs("docs/assets", exist_ok=True)

# Load data
ratings = pd.read_csv("data/raw/ratings.csv")

# Rating Distribution Plot
plt.figure(figsize=(10, 6))
sns.histplot(ratings['rating'], bins=10, kde=False, color='skyblue')
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('docs/assets/rating_distribution.png')
print("Plot saved to docs/assets/rating_distribution.png")
