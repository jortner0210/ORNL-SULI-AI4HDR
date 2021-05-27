import io
import os
import shutil
import sys
import pickle
import argparse

import numpy as np

from PIL import Image
from pathlib import Path
from werkzeug.utils import secure_filename
from tqdm import tqdm

from flask import Flask, render_template, jsonify, send_file, request

from backend.Database import queryImageIdx, importAndProcess, getRandomIds, DatabaseNotFound

# ----------------------------------------------
#  APPLICATION DATA
# ----------------------------------------------

DATABASE_NAME      = None
TEMP_LOC           = "./temp"
QUERY_CACHE_DIR    = "./query-cache"
ALLOWED_EXTENSIONS = { "png", "jpg", "jpeg" }

app = Flask(__name__)

# ----------------------------------------------
#  FLASK ROUTES
# ----------------------------------------------

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/view-database')
def viewDatabase():
    return render_template("database.html")


@app.route('/image-upload')
def imageUpload():
    return render_template("image-upload.html")


@app.route("/get-nearest-neighbors/<int:idx_id>/<int:query_size>")
def getNearestNeighbors(idx_id, query_size):
    '''
    Queries the database for the nearest neighbors for a given image.
    Queries are ordered by relevance: i.e. most relevant first.
    The first element in the returned list is the query image itself.

    :param idx_id:
        Image's autoincremented unique integer id.

    :param query_size:
        Number of images to return in the query.
    '''
    print(f"Querying image: id={idx_id}")

    cache_dir = Path(QUERY_CACHE_DIR)
    if not cache_dir.is_dir():
        print("Initializing Cache")
        try: 
            os.mkdir(cache_dir)
        except:
            print("Cache already created")

    query_cache_filename = cache_dir.joinpath(f"query_cache_{idx_id}")

    nearest = None
    if not query_cache_filename.is_file():
        nearest = queryImageIdx(idx_id, DATABASE_NAME, query_size)
        cache_outfile = open(query_cache_filename, "wb")
        pickle.dump(nearest, cache_outfile)
        cache_outfile.close()
    else:
        cache_infile = open(query_cache_filename, "rb")
        nearest = pickle.load(cache_infile)
        cache_infile.close()

    print(nearest)

    return jsonify([
        nearest
    ])


@app.route("/download-image/<string:db_id>", methods=["POST", "GET"])
def downloadImage(db_id):
    '''
    Send an image from the database.

    :param db_id:
        Database image id
    '''
    db_id = Path(db_id.replace("_", "/"))
    
    arr = np.array(Image.open(db_id))
    
    # convert numpy array to PIL Image
    img = Image.fromarray(arr.astype("uint8"))
    
    # create file-object in memory
    fileObject = io.BytesIO()
    img.save(fileObject, "JPEG")

    # move to beginning of file so `send_file()` will read from start
    fileObject.seek(0)

    return send_file(fileObject, mimetype="image/jpg")


@app.route("/get-random/<int:random_count>", methods=["POST", "GET"])
def getRandom(random_count):
    '''
    Creates and sends a file object for given file
    
    :params random_count:
        Number of random ids to generate.
    '''
    random_ids = getRandomIds(DATABASE_NAME, random_count)
    return jsonify(random_ids)


@app.route("/image-upload", methods=[ "POST", "GET" ])
def uploadImage():
    '''
    Uploads an image from the interface to the database
    '''
    cache_dir = Path(QUERY_CACHE_DIR)
    if cache_dir.is_dir():
        shutil.rmtree(cache_dir)

    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and fileTypeAllowed(file.filename, ALLOWED_EXTENSIONS):           
            filename = secure_filename(file.filename)
            tmpDir   = Path(TEMP_LOC).joinpath(filename.split(".")[0])

            # Add id to folder name to avoid collisions
            ids = [int(f.name.split("-")[0]) for f in Path(TEMP_LOC).iterdir() ]
            uId = 0  
            if len(ids) != 0:
                uId = max(ids) + 1 

            # Save image to temporary dir
            tmpDir = tmpDir.parent.joinpath("{}-{}".format(uId, tmpDir.name))
            os.mkdir(tmpDir)
            file.save(os.path.join(str(tmpDir), filename))

            # Launch import and process
            importAndProcess(str(tmpDir.joinpath(filename)), DATABASE_NAME, "testimage")

            shutil.rmtree(str(tmpDir))
            

    return render_template("image-upload.html")


# ----------------------------------------------
#  HELPER FUNCTIONS
# ----------------------------------------------

def fileTypeAllowed(file_name, allowed_exts):
    '''
    Ensures that a file type is allowed

    :params file_name:
        File name to check

    :params allowed_exts:
        List of allowed extensions

    '''
    return '.' in file_name and file_name.rsplit('.', 1)[1].lower() in allowed_exts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_loc', default=None, help="Database Location.")

    args = parser.parse_args()

    DATABASE_NAME = args.db_loc

    if DATABASE_NAME is None:
        raise DatabaseNotFound("Please provide a database location")
    else:
        app.run(debug=True, host="localhost", port=8000)
    
