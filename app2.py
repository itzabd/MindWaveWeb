from flask import render_template
from flask_login import login_required

from app import app, role_required

import pandas as pd
import matplotlib.pyplot as plt

@app.route('/admin/visual', methods=['GET'])
@login_required
@role_required('admin')
def visual():
    return render_template('visual.html')