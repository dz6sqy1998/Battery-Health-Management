from flask import jsonify ,Flask,request,url_for,redirect
from flask import render_template
import config
app = Flask (__name__)
categories=["衬衫","羊毛衫","雪纺衫","裤子","高跟鞋","袜子"]
data=[5, 20, 36, 10, 10, 20]
app.config.from_object(config)

@app.route("/")
def home():

	return render_template('login_1.html')
# @app.route('/', methods=["GET"])
# def index_1():
# 	if request=='GET':
# 		return redirect(url_for('index_1'))
# 		# return render_template("index_1.html")
# 	return render_template("index.html")

@app.route('/echarts', methods=["GET"]) #echarts 名字可以改为任意，但一定要与HTML文件中一至
def echarts():
    return jsonify(categories = categories,data =data)
if __name__ == '__main__':
	app.run(debug=True,port=200)

