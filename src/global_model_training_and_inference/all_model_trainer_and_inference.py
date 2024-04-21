import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import os

#########################################################################################################################

"""
Contributor- Ekaghni Mukherjee
Date- 23.02.2024
If you want to train/ infer using Cpu then use the ModelTrainerCPU class (Slower Training)
If you want to train/ infer using Gpu then use the ModelTrainerGPU class (Faster Training)
For training/ inferencing using gpu make sure that you have cuda, cudnn and pytorch , tensorflow gpu compatible versions set up on your machine
"""

#################################### CPU Training And Inference ######################################################### 

class ModelTrainerCPU:

    def train_cpu_memory_usage_random_forest_classifier():
        df = pd.read_csv('output.csv')
        X = df[['cpu_usage', 'memory_usage']]
        y = df['signal']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy',n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        joblib.dump(best_model, 'best_model.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        train_pred = best_model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, train_pred)
        print(f"Training Accuracy: {train_accuracy:.2%}")
        test_pred = best_model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, test_pred)
        print(f"Test Accuracy: {test_accuracy:.2%}")
        # feature_importances = best_model.feature_importances_
        # feature_names = X.columns
        # plt.barh(feature_names, feature_importances)
        # plt.xlabel('Feature Importance')
        # plt.title('Feature Importances')
        # plt.show()


    def inference_cpu_memory_usage_random_forest_classifier(cpu_usage: int, memory_usage: int):
        loaded_model = joblib.load('/home/telaverge/process_anomaly_package/src/global_model_training_and_inference/best_model.joblib')
        scaler = joblib.load('/home/telaverge/process_anomaly_package/src/global_model_training_and_inference/scaler.joblib')
        new_data = pd.DataFrame({'cpu_usage': [cpu_usage], 'memory_usage': [memory_usage]})
        scaled_new_data = scaler.transform(new_data)
        prediction = loaded_model.predict(scaled_new_data)
        # print(f"Prediction: {prediction[0]}")
        pred = (str)(prediction[0])
        return  pred


    def train_common_time_series_analysis_lstm():
        data = pd.read_csv("/home/telaverge/process_anomaly_package/src/global_model_training_and_inference/cpu_memory_usage.csv")
        data['Timestamp_of_each_day'] = pd.to_datetime(data['Timestamp_of_each_day'])
        data = data.sort_values(by='Timestamp_of_each_day')
        scaler = MinMaxScaler(feature_range=(0, 1))
        data['Cpu_usage_scaled'] = scaler.fit_transform(data['Cpu_usage'].values.reshape(-1, 1))
        seq_length = 10
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data['Cpu_usage_scaled'].iloc[i:i+seq_length].values)
                y.append(data['Cpu_usage_scaled'].iloc[i+seq_length])
            return np.array(X), np.array(y)
        X, y = create_sequences(data, seq_length)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        X_tensor = X_tensor.view(-1, seq_length, 1)
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)
            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])
                return out
        input_size = 1
        hidden_size = 3
        output_size = 1
        num_epochs = 5
        batch_size = 16
        model = LSTMModel(input_size, hidden_size, output_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Training started...")
        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(-1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}")
        print("Training completed.")
        torch.save(model.state_dict(), 'cpu_usage_model.pt')
        print("Model saved.")
        def is_high_usage(timestamp, high_usage_ranges):
            timestamp = pd.Timestamp(timestamp)
            for start, end in high_usage_ranges:
                if start <= timestamp <= end:
                    return True
            return False
        model_inference = LSTMModel(input_size, hidden_size, output_size)
        model_inference.load_state_dict(torch.load('cpu_usage_model.pt', map_location=torch.device('cpu')))
        model_inference.eval()
        predicted_cpu_usage = model_inference(X_tensor).cpu().detach().numpy()
        threshold = 0.8
        high_usage_indices = np.where(predicted_cpu_usage > threshold)[0]
        high_usage_ranges = []
        for idx in high_usage_indices:
            start_timestamp = data['Timestamp_of_each_day'].iloc[idx]
            end_timestamp = data['Timestamp_of_each_day'].iloc[idx + seq_length - 1]
            high_usage_ranges.append((start_timestamp, end_timestamp))
        timestamp_to_check = "2024-12-31 12:30:00"
        is_in_high_usage = is_high_usage(timestamp_to_check, high_usage_ranges)
        if is_in_high_usage:
            print("Yes")
        else:
            print("No")


    def inference_common_time_series_analysis_lstm(timestamp_to_check: str):
        data = pd.read_csv("/home/telaverge/process_anomaly_package/src/global_model_training_and_inference/cpu_memory_usage.csv")
        data['Timestamp_of_each_day'] = pd.to_datetime(data['Timestamp_of_each_day'])
        data = data.sort_values(by='Timestamp_of_each_day')
        scaler = MinMaxScaler(feature_range=(0, 1))
        data['Cpu_usage_scaled'] = scaler.fit_transform(data['Cpu_usage'].values.reshape(-1, 1))
        seq_length = 10
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data['Cpu_usage_scaled'].iloc[i:i+seq_length].values)
                y.append(data['Cpu_usage_scaled'].iloc[i+seq_length])
            return np.array(X), np.array(y)
        X, y = create_sequences(data, seq_length)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        X_tensor = X_tensor.view(-1, seq_length, 1)
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)
            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])
                return out
        input_size = 1
        hidden_size = 3
        output_size = 1
        num_epochs = 5
        batch_size = 16
        def is_high_usage(timestamp, high_usage_ranges):
            timestamp = pd.Timestamp(timestamp)
            for start, end in high_usage_ranges:
                if start <= timestamp <= end:
                    return True
            return False
        model_inference = LSTMModel(input_size, hidden_size, output_size)
        model_inference.load_state_dict(torch.load('/home/telaverge/process_anomaly_package/src/global_model_training_and_inference/cpu_usage_model.pt', map_location=torch.device('cpu')))
        model_inference.eval()
        predicted_cpu_usage = model_inference(X_tensor).cpu().detach().numpy()
        threshold = 0.8
        high_usage_indices = np.where(predicted_cpu_usage > threshold)[0]
        high_usage_ranges = []
        for idx in high_usage_indices:
            start_timestamp = data['Timestamp_of_each_day'].iloc[idx]
            end_timestamp = data['Timestamp_of_each_day'].iloc[idx + seq_length - 1]
            high_usage_ranges.append((start_timestamp, end_timestamp))
        is_in_high_usage = is_high_usage(timestamp_to_check, high_usage_ranges)

        if is_in_high_usage:
            # print(f"Timestamp---> {timestamp_to_check} Yes")
            return "yes"
        else:
            # print(f"Timestamp---> {timestamp_to_check} No")
            return "no"

######################################################## GPU Training And Inference ###################################### 

class ModelTrainerGPU:

    def train_cpu_memory_usage_random_forest_classifier():
        df = pd.read_csv('output.csv')
        df['signal'] = df['signal'].map({'yes': 1, 'no': 0})
        X = df[['cpu_usage', 'memory_usage']].values
        y = df['signal'].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        class NeuralNetwork(nn.Module):
            def __init__(self, input_size):
                super(NeuralNetwork, self).__init__()
                self.fc1 = nn.Linear(input_size, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 1)
                self.sigmoid = nn.Sigmoid()
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.sigmoid(self.fc3(x))
                return x
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_size = X_train.shape[1]
        model = NeuralNetwork(input_size).to(device)
        X_train_tensor = X_train_tensor.to(device)
        y_train_tensor = y_train_tensor.to(device)
        X_test_tensor = X_test_tensor.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 500
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        with torch.no_grad():
            model.eval()
            y_pred_tensor = model(X_test_tensor)
            y_pred = (y_pred_tensor >= 0.5).float().cpu().numpy()
        y_pred = y_pred.flatten()
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Test Accuracy: {accuracy:.2%}')
        torch.save(model.state_dict(), 'pytorch_model.pth')
        joblib.dump(scaler, 'scaler.joblib')


    def inference_cpu_memory_usage_random_forest_classifier(cpu_usage: int, memory_usage: int):
        class NeuralNetwork(nn.Module):
            def __init__(self, input_size):
                super(NeuralNetwork, self).__init__()
                self.fc1 = nn.Linear(input_size, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 1)
                self.sigmoid = nn.Sigmoid()
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.sigmoid(self.fc3(x))
                return x
        input_size = 2 
        model = NeuralNetwork(input_size)
        model.load_state_dict(torch.load('pytorch_model.pth'))
        scaler = joblib.load('scaler.joblib')
        input_data = scaler.transform([[cpu_usage, memory_usage]])
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = input_tensor.to(device)
        model = model.to(device)
        with torch.no_grad():
            model.eval()
            output_tensor = model(input_tensor)
            print("Output safetensors----> ",output_tensor.item())
            prediction = 'yes' if output_tensor.item() >= 0.8 else 'no'
        print("Prediction:", prediction)
        return prediction


    def train_common_time_series_analysis_lstm():
        data = pd.read_csv("cpu_memory_usage.csv")
        data['Timestamp_of_each_day'] = pd.to_datetime(data['Timestamp_of_each_day'])
        data = data.sort_values(by='Timestamp_of_each_day')
        scaler = MinMaxScaler(feature_range=(0, 1))
        data['Cpu_usage_scaled'] = scaler.fit_transform(data['Cpu_usage'].values.reshape(-1, 1))
        seq_length = 10
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data['Cpu_usage_scaled'].iloc[i:i+seq_length].values)
                y.append(data['Cpu_usage_scaled'].iloc[i+seq_length])
            return np.array(X), np.array(y)
        X, y = create_sequences(data, seq_length)
        X_tensor = torch.tensor(X, dtype=torch.float32).cuda()
        y_tensor = torch.tensor(y, dtype=torch.float32).cuda()
        X_tensor = X_tensor.view(-1, seq_length, 1).cuda()
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)
            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])
                return out
        input_size = 1
        hidden_size = 3
        output_size = 1
        num_epochs = 5
        batch_size = 16
        model = LSTMModel(input_size, hidden_size, output_size).cuda()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Training started...")
        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(-1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}")
        print("Training completed.")
        torch.save(model.state_dict(), 'cpu_usage_model.pt')
        print("Model saved.")
        def is_high_usage(timestamp, high_usage_ranges):
            timestamp = pd.Timestamp(timestamp)
            for start, end in high_usage_ranges:
                if start <= timestamp <= end:
                    return True
            return False
        model_inference = LSTMModel(input_size, hidden_size, output_size).cuda()
        model_inference.load_state_dict(torch.load('cpu_usage_model.pt'))
        model_inference.eval()
        predicted_cpu_usage = model_inference(X_tensor).cpu().detach().numpy()
        threshold = 0.8
        high_usage_indices = np.where(predicted_cpu_usage > threshold)[0]
        high_usage_ranges = []
        for idx in high_usage_indices:
            start_timestamp = data['Timestamp_of_each_day'].iloc[idx]
            end_timestamp = data['Timestamp_of_each_day'].iloc[idx + seq_length - 1]
            high_usage_ranges.append((start_timestamp, end_timestamp))
        timestamp_to_check = "2024-12-31 12:30:00"
        is_in_high_usage = is_high_usage(timestamp_to_check, high_usage_ranges)
        if is_in_high_usage:
            print("Yes")
        else:
            print("No")

    
    def inference_common_time_series_analysis_lstm(timestamp_to_check: str):
        data = pd.read_csv("demo_excel.csv")
        data['Timestamp_of_each_day'] = pd.to_datetime(data['Timestamp_of_each_day'])
        data = data.sort_values(by='Timestamp_of_each_day')
        scaler = MinMaxScaler(feature_range=(0, 1))
        data['Cpu_usage_scaled'] = scaler.fit_transform(data['Cpu_usage'].values.reshape(-1, 1))
        seq_length = 10
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data['Cpu_usage_scaled'].iloc[i:i+seq_length].values)
                y.append(data['Cpu_usage_scaled'].iloc[i+seq_length])
            return np.array(X), np.array(y)
        X, y = create_sequences(data, seq_length)
        X_tensor = torch.tensor(X, dtype=torch.float32).cuda()
        y_tensor = torch.tensor(y, dtype=torch.float32).cuda()
        X_tensor = X_tensor.view(-1, seq_length, 1).cuda()
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)
            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])
                return out
        input_size = 1
        hidden_size = 3
        output_size = 1
        num_epochs = 5
        batch_size = 16
        def is_high_usage(timestamp, high_usage_ranges):
            timestamp = pd.Timestamp(timestamp)
            for start, end in high_usage_ranges:
                if start <= timestamp <= end:
                    return True
            return False
        model_inference = LSTMModel(input_size, hidden_size, output_size).cuda()
        model_inference.load_state_dict(torch.load('cpu_usage_model.pt'))
        model_inference.eval()
        predicted_cpu_usage = model_inference(X_tensor).cpu().detach().numpy()
        threshold = 0.8
        high_usage_indices = np.where(predicted_cpu_usage > threshold)[0]
        high_usage_ranges = []
        for idx in high_usage_indices:
            start_timestamp = data['Timestamp_of_each_day'].iloc[idx]
            end_timestamp = data['Timestamp_of_each_day'].iloc[idx + seq_length - 1]
            high_usage_ranges.append((start_timestamp, end_timestamp))
        
        is_in_high_usage = is_high_usage(timestamp_to_check, high_usage_ranges)

        if is_in_high_usage:
            print(f"Timestamp---> {timestamp_to_check} Yes")
            return "yes"
        else:
            print(f"Timestamp---> {timestamp_to_check} No")
            return "no"



