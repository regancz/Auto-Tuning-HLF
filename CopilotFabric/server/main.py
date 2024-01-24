from flask import Flask
from flask_cors import CORS

from CopilotFabric.server.api.dashboard import dashboard_api
from CopilotFabric.server.api.optimize import optimize_api

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.register_blueprint(dashboard_api, url_prefix='/api/dashboard')
app.register_blueprint(optimize_api, url_prefix='/api/optimize')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
