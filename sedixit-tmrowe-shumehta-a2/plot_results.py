import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(training_metrics_filename, test_metrics_filename, path):
    df_train = pd.read_csv(path + training_metrics_filename)
    df_test = pd.read_csv(path + test_metrics_filename)
    
    # Training and Validation Loss Plot
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    for model in df_train['Model'].unique():
        subset = df_train[df_train['Model'] == model]
        plt.plot(subset['Epoch'].to_numpy(), subset['Train Loss'].to_numpy(), label=f'{model} Train Loss', marker='o')
        plt.plot(subset['Epoch'].to_numpy(), subset['Validation Loss'].to_numpy(), label=f'{model} Validation Loss', marker='x')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    training_loss_plot_filename = path + 'training_validation_loss.png'
    plt.savefig(training_loss_plot_filename)
    plt.close() 

    # Training and Validation Accuracy Plot
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 2)
    for model in df_train['Model'].unique():
        subset = df_train[df_train['Model'] == model]
        plt.plot(subset['Epoch'].to_numpy(), subset['Train Accuracy'].to_numpy(), label=f'{model} Train Accuracy', marker='o')
        plt.plot(subset['Epoch'].to_numpy(), subset['Validation Accuracy'].to_numpy(), label=f'{model} Validation Accuracy', marker='x')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    training_accuracy_plot_filename = path + 'training_validation_accuracy.png'
    plt.savefig(training_accuracy_plot_filename)
    plt.close()

    # Final Test Accuracy Plot
    plt.figure(figsize=(10, 6))
    models = df_test['Model'].values
    accuracies = df_test['Test Accuracy'].values
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    plt.bar(models, accuracies, color=colors)
    plt.title('Final Test Accuracy by Model')
    plt.xlabel('Model')
    plt.ylabel('Test Accuracy (%)')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    test_accuracy_plot_filename = path + 'final_test_accuracy.png'
    plt.savefig(test_accuracy_plot_filename)
    plt.close()

plot_metrics('model_metrics.csv', 'test_metrics.csv', '/Users/tylerrowe/Desktop/College/24Spring/Computer Vision/Assignments/sedixit-tmrowe-shumehta-a2/model_results/')
