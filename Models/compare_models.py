import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

training = "Data/train.csv"
test = "Data/test.csv"

def load_data():
    print("Loading dataset")

    training_df = pd.read_csv(training)
    test_df = pd.read_csv(test)

    X_train = training_df.drop(['subject', 'Activity'], axis = 1)
    y_train = training_df['Activity']

    X_test = test_df.drop(['subject', 'Activity'], axis=1)
    y_test = test_df['Activity']

    return X_train, X_test, y_train, y_test

def train_and_eval():
    X_train, X_test, y_train, y_test = load_data()

    #Scaling data
    #If data isnt scaled between -1 and 1 neutrla network behave poorly
    print("\nData scaling")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #SVM model (Support Vector machine)
    print("\nTraining SVM model")
    model_svm = SVC(kernel='linear', random_state=42)
    model_svm.fit(X_train_scaled, y_train)
    pred_svm = model_svm.predict(X_test_scaled)
    acc_svm = accuracy_score(y_test, pred_svm)
    print(f"SVM Accuracy: {acc_svm:.4f}")

    # NN (MLP) model
    print("Training Neural Network (MLP) model")
    model_mlp = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=50, random_state=42)
    model_mlp.fit(X_train_scaled, y_train)
    pred_mlp = model_mlp.predict(X_test_scaled)
    acc_mlp = accuracy_score(y_test, pred_mlp)
    print(f"Accuracy of Neural Network model: {acc_mlp:.4f}")

    print("-------Conclusion-------")
    if acc_mlp > acc_svm:
        print(f"Neural network accuracy is better (+{(acc_mlp - acc_svm)*100:.2f}%)")
    else:
        print(f"SVM accuracy is better (+{(acc_svm - acc_mlp)*100:.2f}%)")
    
    models = ['SVM', 'Neural Network']
    accuracies = [acc_svm, acc_mlp]
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=models, y=accuracies, hue=models, legend=False, palette='viridis')
    plt.ylim(0.9, 1.0) 
    plt.title('Model Comparison: SVM vs Neural Network')
    plt.ylabel('Accuracy')
    plt.show()

if __name__ == "__main__":
    train_and_eval()