from src.components.data_ingestion import load_data
from src.components.data_transformation import get_preprocessor
from src.components.model_trainer import train_model

def run_training():
    df = load_data("Data/student_dataset.csv")

    X = df.drop("Final_Exam_Score", axis=1)
    y = df["Final_Exam_Score"]

    preprocessor = get_preprocessor()

    train_model(X, y, preprocessor)

    print("Training complete!")

if __name__ == "__main__":
    run_training()