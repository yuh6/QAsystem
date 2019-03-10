#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, render_template
import answer

app = Flask(__name__)



@app.route('/', methods=['GET'])
def signin_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def signin():
    question = request.form['question']
    #if username=='admin' and password=='password':
    answer = q.answer_question(question, 0)
    return render_template('signin-ok.html', username=answer)
    #return render_template('form.html', message='Bad username or password', username=username)

if __name__ == '__main__':
    app.run()
