
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def custom_select_page():
    return render_template('test_server.html')

@app.route('/process')
def process_form():
    select_option = request.args.get('select_option')

    # Process the form input as needed

    return "Received select option value: " + select_option

if __name__ == '__main__':
    app.run()
