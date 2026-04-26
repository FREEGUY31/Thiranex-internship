import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Style
sns.set(style="whitegrid")

# Load dataset
df = pd.read_csv("Sales Dataset.csv")

# -----------------------------
# Data Cleaning
# -----------------------------

# Convert date column
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Remove duplicates
df = df.drop_duplicates()

# -----------------------------
# KPIs
# -----------------------------
total_sales = df['Amount'].sum()
total_profit = df['Profit'].sum()
total_orders = df['Order ID'].nunique()
avg_sales = df['Amount'].mean()

print("Total Sales:", total_sales)
print("Total Profit:", total_profit)
print("Total Orders:", total_orders)
print("Average Sales:", avg_sales)

# -----------------------------
# 1. Sales Over Time
# -----------------------------
sales_trend = df.groupby('Order Date')['Amount'].sum()

plt.figure()
sales_trend.plot()
plt.title("Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.show()

# -----------------------------
# 2. Sales by Category
# -----------------------------
plt.figure()
sns.barplot(x='Category', y='Amount', data=df, estimator=sum)
plt.title("Sales by Category")
plt.xticks(rotation=30)
plt.show()

# -----------------------------
# 3. Top 10 Sub-Categories
# -----------------------------
top_subcat = df.groupby('Sub-Category')['Amount'].sum().sort_values(ascending=False).head(10)

plt.figure()
top_subcat.plot(kind='bar')
plt.title("Top 10 Sub-Categories")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.show()

# -----------------------------
# 4. State-wise Sales
# -----------------------------
state_sales = df.groupby('State')['Amount'].sum().sort_values(ascending=False).head(10)

plt.figure()
state_sales.plot(kind='bar')
plt.title("Top 10 States by Sales")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.show()

# -----------------------------
# 5. Payment Mode Distribution
# -----------------------------
payment = df['PaymentMode'].value_counts()

plt.figure()
payment.plot(kind='pie', autopct='%1.1f%%')
plt.title("Payment Mode Distribution")
plt.ylabel("")
plt.show()