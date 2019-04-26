from flask import Flask, jsonify, request, make_response, flash, url_for, redirect
from flask_httpauth import HTTPBasicAuth
from API import ITRT
import logging

app = Flask(__name__)
auth = HTTPBasicAuth()
api_list = ('ITRT','TS','SDA','GAP')
ver_list = ('1.0')

handler = logging.FileHandler('app2.log', encoding='UTF-8')
app.logger.addHandler(handler)

@auth.get_password
def get_password(username):
    if username == 'Demo':
        return 'Demo'
    return None

@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({'error': 'Bad request'}), 400)

@auth.error_handler
def unauthorized():
    return make_response(jsonify({'error': 'Unauthorized access'}), 401)

@app.before_request
def before_request():
    app.logger.info("before_request,")
    app.logger.info(request.headers)
    #return redirect(url_for('task_list'))

@app.route('/daas', methods=['GET'])
#@auth.login_required
def task_list():
    app.logger.info("In Get,")

    return "Your task list:"

@app.route('/daas', methods=['POST'])
#@auth.login_required
def task_admin():
    app.logger.info("In Post,")
    return 'post'
    if(not ValidRequest(request.json)):
        abort(400)
    api = request.json['api']
    version = request.json['version']
    content = request.json['content']
    if api=='ITRT':
        service = ITRT()
        result = service.Execute(version, content)
        UpdateDB(result)
        app.logger.info(result)
        return jsonify({'response': result}), 201

    elif api=='TS':
        return 'TS TBD'

    elif api=='SDA':
        return 'SDA TBD'

    else:
        return 'GAP TBD'

def ValidRequest(r_json):
    #r_json = r_json
    if (not r_json or 'api' not in r_json or 'version' not in r_json or 'content' not in r_json):
        return False
    
    api = r_json['api']
    version = r_json['version']
    if ( api not in api_list or version not in ver_list) :
        return False

    else:
        return True

def UpdateDB(result):
    pass

if __name__ == '__main__':
    app.run(debug=True, port=int(8080))
