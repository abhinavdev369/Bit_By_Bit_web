import matplotlib.pyplot as plt
import os

# Ensure the static directory exists
static_dir = 'c:/proj1/static/'
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# Waste Trends 2024
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plastic = [8, 9, 10, 11, 12, 13, 14, 15, 14, 13, 12, 11]
shoes = [5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 6]
metal = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]

plt.figure(figsize=(10, 6))
plt.plot(months, plastic, label='Plastic (tons)', marker='o')
plt.plot(months, shoes, label='Shoes (tons)', marker='o')
plt.plot(months, metal, label='Metal (tons)', marker='o')
plt.title('Monthly Waste Generation Trends (2024)')
plt.xlabel('Month')
plt.ylabel('Waste (tons)')
plt.legend()
plt.grid(True)
plt.savefig('c:/proj1/static/waste_trends_2024.jpg')
plt.close()

# Waste Reduction Impact
categories = ['Plastic', 'Shoes', 'Metal']
before = [12, 9, 8]
after = [9, 7, 6]

x = range(len(categories))
plt.figure(figsize=(8, 6))
plt.bar(x, before, width=0.4, label='Before', align='center')
plt.bar([i + 0.4 for i in x], after, width=0.4, label='After', align='center')
plt.xticks([i + 0.2 for i in x], categories)
plt.title('Impact of Recycling Measures (2024)')
plt.ylabel('Waste (tons)')
plt.legend()
plt.savefig('c:/proj1/static/waste_reduction_2024.jpg')
plt.close()

print("Graphs saved successfully to c:/proj1/static/")