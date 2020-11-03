from flask import Flask, render_template, request, send_file, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from io import BytesIO

import downloadData
import FPGrowth_DNN
import datapreprocess

app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite://///Users/yixian/PycharmProjects/flaskyooo/FYPDatabase.db'
db = SQLAlchemy(app)

class FileContents(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  name = db.Column(db.String(300))
  data = db.Column(db.LargeBinary)

# we clear all the file here when we start the program
db.drop_all()
db.create_all()
file = open("answer.txt","r+")
file. truncate(0)
file. close()
file = open("dataFile.data","r+")
file. truncate(0)
file. close()
file = open("preData.txt","r+")
file. truncate(0)
file. close()

ALLOWED_EXTENSIONS = {'txt', 'data', 'csv'}
def allowed_file(filename):
    """
    This function is use to check whether the file format is correct
    :param filename: the file that user input
    :return: true if is the correct format, false if is the wrong format
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")

def home():
    """
    This function will clean all the data in the database.
    Cleaning the database will make sure the previous dataset won't be inside the database
    :return: the homepage HTML file
    """
    db.drop_all()
    db.create_all()
    return render_template("aboutUs.html")

@app.route("/whatonearth")
def mainPage():
    """
    :return: the run algo page
    """
    return render_template("home.html")

@app.route("/instruction")
def instruction():
    """
    :return: the first page of introduction page
    """
    return render_template("instruction.html")

@app.route("/instruction2")
def instruction2():
    """
    :return: the second page of introduction page
    """
    return render_template("instruction2.html")

@app.route("/instruction3")
def instruction3():
    """
    :return: the third page of introduction page
    """
    return render_template("instruction3.html")

@app.route('/upload', methods=['GET','POST'])
def upload():
    """
    When the upload file picture is click, we will ask user to input the file
    If the input file is submitted, it will download the file to the database
    and flash upload succesfully to the user
    else we will flash user that no file uploaded.
    return: the run algo page
    """
    if request.method == "POST":
        file = request.files['inputFile']
        if file.filename == '':
            flash("No file uploaded")

        elif allowed_file(file.filename):
            db.drop_all()
            db.create_all()
            newFile = FileContents(name=file.filename, data=file.read())
            db.session.add(newFile)
            db.session.commit()
            downloadData.downloadFile('http://127.0.0.1:5000/download')
            flash("Uploaded Successfully!")
        else:
            flash("Wrong File Format")

        return redirect(request.url)

    return render_template("home.html")

@app.route('/download')
def download():
    """
    Download the dataset content into the dataset.csv
    :return: the content of the dataset??
    """
    file_data = FileContents.query.filter_by(id=1).first()
    return send_file(BytesIO(file_data.data),attachment_filename='dataset.csv',as_attachment=True)
    # return send_file("answer.txt",attachment_filename='dataset.csv',as_attachment=True)

@app.route('/askdata',methods=["POST","GET"])
def dataInput():
    """
    This function will retrieve the data input from the user
    we will ask user for class column, class name, support value, confidence value,
    numeric column and preprocessonly checkbox
    If the preprocessonly check box is not tick, we will call the runalgo function
    If the preprocessonly check box is tick, we will only run our preprocess function
    :return: redirect it to runAlgorithm() method
    """
    if request.method == "POST":
        classCol = request.form["nm"]
        className = request.form["attributeName"]
        supportValue = request.form["support"]
        confidenceValue = request.form["confidence"]
        if className == "" or className == "none":
            className = "empty"
        nmCol = request.form["numericalCol"]
        if nmCol == "" or nmCol == "none":
            nmCol = "empty"
        if request.form.get("preOnly"):
            return redirect(url_for("runAlgorithm", value=classCol, nmColList=nmCol, preprocessOnly= True, column_name = className, support_value = supportValue, confidence_value = confidenceValue))
        else:
            return redirect(url_for("runAlgorithm", value=classCol, nmColList=nmCol, preprocessOnly= False, column_name = className, support_value = supportValue, confidence_value = confidenceValue))

    else:
        return render_template("dataclarify.html")

@app.route('/run/<value>/<nmColList>/<preprocessOnly>/<column_name>/<support_value>/<confidence_value>')
def runAlgorithm(value, nmColList, preprocessOnly, column_name, support_value, confidence_value):
    """
    In this function we preprocess the data that the user give to us.
    we will make the class name data, numeric column data into a list in order to let the code to run
    If the user only want preprocess data, we will only run preprocess data and display to user
    If the user want to use our algo, we will run our algo and display the result to user
    :param value: the class column
    :param nmColList: the numeric column
    :param preprocessOnly: ask user want preprocess only or not
    :param column_name: the column name for each attribute
    :param support_value: the support value that the user want
    :param confidence_value: the confidence value that the user want
    :return: the result of the algo or the result of the preprocess data in a page
    """
    # fpgrowth.runalgolo()
    classColumn = value
    numerical_Col = nmColList
    col_name= column_name
    li = None
    col_li = None
    if nmColList != "empty":
        li = list(numerical_Col.split(","))
        for i in range(len(li)):
            li[i] = int(li[i])
    if column_name != "empty":
        col_li = list(column_name.split(","))

    if preprocessOnly == "True":
        preData = datapreprocess.preprocess_data("dataFile.data", numeric_columns=li, column_names=col_li,)

        string = "Preprocessed Data is here!"

        f = open("preData.txt", "w")
        f.write(str(string) + "\n")
        f.close()
        for i in range(len(preData)):
            string = preData[i]

            f = open("preData.txt", "a")
            f.write(str(string) + "\n")
            f.close()
        with open("preData.txt", "r") as f:
            content = f.read()
        return render_template("content.html", content=content)
    else:
        try:
            FPGrowth_DNN.runAC(classColumn,int(support_value),float(confidence_value),li,col_li)
        except Exception as e:
            return render_template("error.html", error=str(e))

        # FPGrowth_DNN.runalgolo(classColumn, li, col_li)

        with open("answer.txt", "r") as f:
            content = f.read()
            return render_template("content.html", content=content)

@app.route('/result')
def result():
    """
    Create a file and write all the result into the file
    :return: a result text file to the user
    """
    return send_file("answer.txt",attachment_filename='result.txt',as_attachment=True)


if __name__ == "__main__":
    app.secret_key = "is a secret"
    app.run(debug=True)