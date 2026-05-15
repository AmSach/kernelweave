#!/usr/bin/env python3
from flask import Flask, jsonify

app = Flask(__name__)

products = [
    {"id": 1, "name": "Quantum Computer", "price": 50000},
    {"id": 2, "name": "Superconducting Chip", "price": 1000},
    {"id": 3, "name": "Quantum Analyzer", "price": 2000}
]

@app.route('/api/products', methods=['GET'])
def get_products():
    return jsonify(products)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)