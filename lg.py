import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def main():
    try:
        # Prompt user for CSV file name
        file_name = input("Enter the CSV file name (with extension): ")
        # Load the CSV file into a DataFrame
        data = pd.read_csv(file_name)
        print("\nColumns in the dataset:\n", list(data.columns))

        # Prompt user for X and Y column names
        x_column = input("\nEnter the column name for the independent variable (X): ")
        y_column = input("Enter the column name for the dependent variable (Y): ")

        # Extract X and Y from the dataset
        X = data[[x_column]]
        Y = data[[y_column]]

        # Split the data into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Initialize and train the linear regression model
        model = LinearRegression()
        model.fit(X_train, Y_train)

        # Predict on the test set
        Y_pred = model.predict(X_test)

        # Output model parameters and evaluation metrics
        print("\nModel Coefficients:", model.coef_[0])
        print("Model Intercept:", model.intercept_[0])
        print("Mean Squared Error:", mean_squared_error(Y_test, Y_pred))
        print("R^2 Score:", r2_score(Y_test, Y_pred))

        # Prompt user for a prediction
        while True:
            try:
                new_value = float(input("\nEnter a value for prediction (or 'q' to quit): "))
                prediction = model.predict([[new_value]])
                print(f"Predicted value: {prediction[0][0]:.2f}")
            except ValueError:
                print("Exiting prediction loop.")
                break
    except FileNotFoundError:
        print("Error: File not found. Please check the file name and try again.")
    except KeyError as e:
        print(f"Error: Column {e} not found in the dataset.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
