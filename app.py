import pickle
import gradio as gr

# Load trained model
with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

flower_names = {
    0: "🌸 Setosa",
    1: "🌺 Versicolor",
    2: "🌼 Virginica"
}

flower_description = {
    0: "Setosa flowers are small and have short petals.",
    1: "Versicolor flowers have medium-sized petals.",
    2: "Virginica flowers are the largest with long petals."
}

def predict_flower(sepal_length, sepal_width, petal_length, petal_width):

    prediction = model.predict([[sepal_length,
                                 sepal_width,
                                 petal_length,
                                 petal_width]])

    probabilities = model.predict_proba([[sepal_length,
                                          sepal_width,
                                          petal_length,
                                          petal_width]])

    confidence = max(probabilities[0]) * 100

    species = flower_names[prediction[0]]

    description = flower_description[prediction[0]]

    return species, f"{confidence:.2f}%", description


app = gr.Interface(
    fn=predict_flower,

    inputs=[
        gr.Slider(4.0,8.0,label="Sepal Length"),
        gr.Slider(2.0,4.5,label="Sepal Width"),
        gr.Slider(1.0,7.0,label="Petal Length"),
        gr.Slider(0.1,3.0,label="Petal Width")
    ],

    outputs=[
        gr.Textbox(label="Predicted Flower"),
        gr.Textbox(label="Confidence"),
        gr.Textbox(label="Description")
    ],

    title="🌸 Iris Flower Classifier",

    description="""
    Predict Iris flower species using Machine Learning.

    Enter flower measurements and click Submit.
    """,

    theme=gr.themes.Soft()
)

app.launch()