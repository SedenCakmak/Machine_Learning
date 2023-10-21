import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Melbourne'deki ev fiyatları veri kümesini yükle
data = pd.read_csv("Melbourne_housing_FULL.csv")

#Eksik Değerleri Kontrol Etme
print(data.isnull().sum())

# Eksik verileri temizle
data = data.dropna(subset=["Price", "YearBuilt", "Rooms", "Suburb", "Address", "Type", "Method", "SellerG", 
                          "Date", "Distance", "Postcode", "Bedroom2", "Bathroom", "Car", "Landsize", 
                          "BuildingArea", "YearBuilt", "CouncilArea", "Lattitude", "Longtitude", 
                          "Regionname", "Propertycount"])

# Label Encoding için bir LabelEncoder nesnesi oluştur
label_encoder = LabelEncoder()

# Kategorik sütunları sayısal değerlere dönüştür
categorical_columns = ["Suburb","Address","Rooms","Type","Price","Method","SellerG","Date","Distance","Postcode","Bedroom2",
                       "Bathroom","Car","Landsize","BuildingArea","YearBuilt","CouncilArea","Lattitude","Longtitude",
                       "Regionname","Propertycount"]
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# Bağımsız değişkenler (X) ve bağımlı değişken (y) ayırma
X = data[["Suburb","Address","Rooms","Type","Method","SellerG","Date","Distance","Postcode","Bedroom2",
          "Bathroom","Car","Landsize","BuildingArea","YearBuilt","CouncilArea","Lattitude","Longtitude",
          "Regionname","Propertycount"]]
y = data['Price']

# Veriyi eğitim ve test setlerine böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge Regresyon modelini oluştur
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Test seti üzerinde tahmin yap
y_pred = model.predict(X_test)

# Model performansını değerlendirme
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Tahminler ile gerçek değerleri karşılaştırarak bir grafik oluştur
plt.scatter(y_test, y_pred,)
plt.xlabel("Gerçek Fiyatlar")
plt.ylabel("Tahmin Edilen Fiyatlar")
plt.title("Gerçek Fiyatlar vs. Tahmin Edilen Fiyatlar")
plt.show()

# Önceden hesaplanmış MSE değeri
mse_value = mse

# Gerçek değerlerin ortalamasını alarak toplam veri noktalarının varyansını hesapla
mean_actual_values = y_test.mean()
total_variance = ((y_test - mean_actual_values) ** 2).mean()

# Doğruluk değerini yüzde olarak hesapla ve virgülden sonra bir basamak olarak yazdır
accuracy_percentage = 100 * (1 - mse / total_variance)
print(f"Accuracy Percentage: {accuracy_percentage:.1f}%")

# MSE değerini yüzde olarak ifade et ve virgülden sonra bir basamak olarak yazdır
mse_percentage = (mse / total_variance) * 100
print(f"Mean Squared Error Percentage: {mse_percentage:.1f}%")
