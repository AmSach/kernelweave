# app.py
from flask import Flask, jsonify

app = Flask(__name__)

products = [
    {"id": 1, "name": "Quantum Processor", "price": 9999},
    {"id": 2, "name": "Superconducting Circuit", "price": 4999},
    {"id": 3, "name": "Entangled Photon Device", "price": 2999}
]

@app.route('/api/products')
def get_products():
    return jsonify(products)

if __name__ == '__main__':
    app.run(debug=True, port=5000)