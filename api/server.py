from flask import Flask,request, jsonify
import artifacts.api.util as util
from flask_cors import CORS


app=Flask(__name__)
CORS(app)

@app.route('/', methods=["GET", "POST"])
def home():
    try:
        print("Server is running")

        # Check if 'image_data' exists in the request
        if 'image_data' not in request.form:
            return jsonify({"error": "Missing 'image_data' in request"}), 400

        # Get the image data
        image_data = request.form['image_data']

        # Use the utility function to classify the image
        response = jsonify(util.classify_image(image_data))
        response.headers.add('Access-Control-Allow-Origin', '*')
        
        print(response)
        return response

    except Exception as e:
        # Return an error response for debugging
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    util.load_saved_artifacts()
    app.run()
