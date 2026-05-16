import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. Load Data
# Ensure your terminal is inside the "week 3" directory where "car data.csv" resides
df = pd.read_csv("B:\\thiranex\\week 3\\car data.csv")

# 2. Feature Selection & Categorical Encoding
# This automatically converts columns like fuel, seller_type, and transmission into numeric flags
features = ['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']
X = pd.get_dummies(df[features], drop_first=True)
y = df['selling_price']

# 3. Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Predictive Model (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make Predictions
y_pred = model.predict(X_test)

# 6. Evaluate the Model
r2 = r2_score(y_test, y_pred)
print("\n--- Model Evaluation ---")
print(f"Model Accuracy (R2 Score): {r2 * 100:.2f}%")

# 7. Visualize Actual vs Predicted Prices
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color='purple', alpha=0.6, ax=ax)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, linestyle='--')
ax.set_title('Actual vs Predicted Car Prices')
ax.set_xlabel('Actual Selling Price')
ax.set_ylabel('Predicted Selling Price')
plt.tight_layout()

# Save the plot directly as an image for your GitHub repository submission
plt.savefig('actual_vs_predicted.png')
plt.close()
print("Visualization saved as 'actual_vs_predicted.png'!")

# 8. Save a sample of predictions to a CSV file
output = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.round(2)})
output.to_csv('predictions_output.csv', index=False)
print("Results successfully saved to 'predictions_output.csv'!")